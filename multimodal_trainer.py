import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

class MultimodalTrainer(Trainer):

    def pairwise_mse_distance(self, codebook):
        # tensor: (token_num, embedding_dim)
        # Calculate squared norms of each row (token)
        # Compute squared norms (shape: [N, 1])
        codebook_norm = (codebook ** 2).sum(dim=1, keepdim=True)
        # Compute pairwise squared distances using the algebraic formula
        img_distance_matrix = codebook_norm + codebook_norm.t() - 2 * torch.mm(codebook, codebook.t())
        # Optionally take the mean over the feature dimension (divide by d)
        img_distance_matrix = img_distance_matrix / codebook.size(-1)
        img_distance_matrix = img_distance_matrix.clamp(min=0.0)
        return img_distance_matrix

    def pairwise_euclidean_distance(self, codebook):
        # tensor: (token_num, embedding_dim)
        # Calculate squared norms of each row (token)
        # Compute squared norms (shape: [N, 1])
        codebook_norm = (codebook ** 2).sum(dim=1, keepdim=True)
        # Compute pairwise squared distances using the algebraic formula
        img_distance_matrix = codebook_norm + codebook_norm.t() - 2 * torch.mm(codebook, codebook.t())
        # Optionally take the mean over the feature dimension (divide by d)
        img_distance_matrix = torch.sqrt(torch.clamp(img_distance_matrix, min=0.0))
        return img_distance_matrix

    def get_model(self):
        if PeftModel and isinstance(self.model, PeftModel):
            return self.model.model
        else:
            return self.model
    
    def compute_shifted_cross_entropy_loss(self, logits, labels):
        """Computes cross-entropy loss with shifted labels."""
        shifted_labels = labels[:, 1:].contiguous().view(-1)  # Shift left
        shifted_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100)
    
    def compute_mse_loss(self, image_embeddings, image_labels):
        return F.mse_loss(image_embeddings, image_labels)
            
    def __init__(self, *args, **kwargs):
        """
        Precompute a distance matrix over image tokens (in BPE space) and
        set up the tensor of BPE indices for image tokens. Here we assume:
        
          - self.model.vocabulary_mapping.image_token_num: number of image tokens (e.g., 1024)
          - self.model.vocabulary_mapping.image_token_ids: a sorted list of BPE token IDs corresponding to image tokens.
          - self.model.vocabulary_mapping.bpe2img: a dict mapping BPE token IDs to their corresponding visual token indices.
          - self.model.codebook: a tensor of shape (image_token_num, D) for visual tokens.
          
        Because the image_token_ids are already sorted, we can directly use them to compute the subspace indices.
        """
        self.image_loss_weight = kwargs.pop("image_loss_weight") if "image_loss_weight" in kwargs else 1.0
        self.text_loss_weight = kwargs.pop("text_loss_weight") if "text_loss_weight" in kwargs else 1.0
        self.sum_loss = kwargs.pop("sum_loss") if "sum_loss" in kwargs else False
        self.loss_type = kwargs.pop("loss_type", "mse")
        self.image_token_index = kwargs.pop("image_token_index") if "image_token_index" in kwargs else 262144
        self.boi_id = kwargs.pop("boi_id", 255999)  # Beginning of Image token ID
        self.eoi_id = kwargs.pop("eoi_id", 256000)
        self.accumulated_loss = dict(ce_loss_text=0.0, regress_loss_images = 0.0, steps=0.0)
        super().__init__(*args, **kwargs)
        
    def update_loss(self, outputs):
        self.accumulated_loss["ce_loss_text"] += outputs["ce_loss_text"]
        self.accumulated_loss["regress_loss_images"] += outputs["regress_loss_images"]
        self.accumulated_loss["steps"] += 1
    
    def clear_loss(self):
        self.accumulated_loss["ce_loss_text"] = 0.0
        self.accumulated_loss["regress_loss_images"] = 0.0
        self.accumulated_loss["steps"] = 0.0

    def _sync_across_processes(self, loss_dict, device):
        return {k: self.accelerator.reduce(torch.tensor(v, device=device, dtype=torch.float32),
                                        reduction="mean").item()
                for k, v in loss_dict.items()}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute total loss as: loss = CE_loss + perceptual_loss
        
        Steps:
         1. Shift logits and labels for autoregressive next-token prediction.
         2. Compute standard cross-entropy loss (CE_loss) over all tokens.
         3. Identify image tokens by checking whether each shifted label is in self.image_bpe_indices.
         4. For each image token:
              a. From the full logits, gather the columns corresponding to self.image_bpe_indices.
              b. Compute softmax probabilities over this image token subspace.
              c. Convert the ground-truth BPE label to its subspace index using torch.searchsorted.
              d. One-hot encode and multiply (via matmul) with self.img_distance_matrix to obtain the “distance row.”
              e. Compute the dot product between the probability distribution and the distance row.
         5. Average the perceptual loss over all image tokens and add it to the CE_loss. 
        """
        if "labels" in inputs:
            labels = inputs["labels"].clone()
        else:
            labels = None
        outputs = model(**inputs, output_hidden_states=True, loss_type=self.loss_type)
        logits = outputs["logits"]         # (batch_size, seq_length, vocab_size)

        # Identify image tokens by checking membership in self.image_bpe_indices.
        image_mask = (labels == self.image_token_index).to(torch.bool).to(labels.device)  # (B, seq_len) bool tensor
        labels[image_mask] = -100  # Mask out image tokens for CE loss.
        
        # Cross-Entropy Loss over the full vocabulary.
        ce_loss = self.compute_shifted_cross_entropy_loss(logits, labels) if self.text_loss_weight != 0 else torch.tensor(0.0, device=labels.device)
        regress_loss_images = torch.tensor(0.0, device=labels.device) if outputs.image_loss is None else outputs.image_loss
        
        total_loss = self.text_loss_weight  * ce_loss + self.image_loss_weight * regress_loss_images
        self.update_loss({"ce_loss_text": ce_loss.detach().cpu().item(),
                          "regress_loss_images": regress_loss_images.detach().cpu().item()})
        log_prefix = "" if self.model.training else "eval_"
        if (self.model.training and self.state.global_step % self.args.logging_steps == 0) or (not self.model.training and self.accelerator.gradient_state.end_of_dataloader):
            synced = self._sync_across_processes(self.accumulated_loss, total_loss.device)

            if self.is_world_process_zero():     
                self.log({f"{log_prefix}ce_loss_text": synced["ce_loss_text"] / synced["steps"],
                          f"{log_prefix}regress_loss_images": synced["regress_loss_images"] / synced["steps"]})
            self.clear_loss()
        if self.model.training:
            total_loss /= self.args.gradient_accumulation_steps
        return (total_loss, outputs) if return_outputs else total_loss
