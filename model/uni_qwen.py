import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLConfig, GenerationConfig
from collections import OrderedDict
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VisionTransformerPretrainedModel, Qwen2RMSNorm
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Union, Tuple, List
from transformers.cache_utils import Cache
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
import torch.nn.functional as F
from transformers.generation.utils import GenerateOutput, logger
from transformers.generation.configuration_utils import CompileConfig
import os
import sys
import time
sys.append = os.path.dirname(os.path.abspath(__file__))
from model.perceiver import PerceiverAR

def wait_for_env(var_name: str, poll_interval_sec: float = 2.0):
    while True:
        val = os.getenv(var_name)
        if val is not None and val.strip() != "":
            return val
        time.sleep(poll_interval_sec)

class Qwen2_5_VLPatchReshape(nn.Module):
    def __init__(self, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x).contiguous().view(-1, self.hidden_size)
        return x
    
class UniQwenVisionTransformer(Qwen2_5_VisionTransformerPretrainedModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.merger = Qwen2_5_VLPatchReshape(config.hidden_size, config.spatial_merge_size)
        self.hidden_size = config.hidden_size

class RegressionHeadPerceiver(nn.Module):
    def __init__(self, text_hidden_size, vision_hidden_size, max_prefix_len = 8192, num_heads=8, dropout=0.1, cross_depth=2, self_septh=8, spatial_merge_size=2, image_seq_len=256, mlp_dim=5120):
        super().__init__()
        self.perceiver = PerceiverAR(
            dim = vision_hidden_size, 
            depth = self_septh, 
            dim_head = vision_hidden_size // num_heads, 
            heads = num_heads, 
            max_seq_len = 32768, 
            cross_attn_seq_len = max_prefix_len, 
            cross_attn_dropout = dropout,
            perceive_depth = cross_depth,
        )
        self.mlp = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, mlp_dim),
        )
        self.dim = mlp_dim
        self.vision_hidden_size = vision_hidden_size

    def forward(self, x, prefix_mask=None, total_mask=None):
        # x: (batch, seq_len, text_hidden_size)
        out = torch.randn(x.shape[0], x.shape[1], self.dim, device=x.device, dtype=x.dtype)
        out[total_mask] = self.mlp(x[total_mask])  # (batch * seq_len, vision_hidden_size)
        assert not torch.isnan(out).any()
        out = out.reshape(out.shape[0], -1, self.vision_hidden_size)
        out = self.perceiver(out, prefix_mask=prefix_mask.repeat_interleave(4, dim=1))
        assert not torch.isnan(out).any()
        return out

@dataclass
class UniQwenOutputWithImageLoss(Qwen2_5_VLCausalLMOutputWithPast):
    """
    Extends Qwen2_5_VLCausalLMOutputWithPast to include image loss and transformed features.

    Args:
        image_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss computed for the image feature regression task.
        transformed_features (`torch.FloatTensor` of shape `(batch_size, vision_hidden_size)`, *optional*):
            Features transformed by the regression head.
    """
    image_loss: Optional[torch.FloatTensor] = None
    transformed_features: Optional[torch.FloatTensor] = None
    
class UniQwenForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    
    Attributes:
        regression_head (nn.Linear): A linear layer that projects features from text hidden dimension
                                   to vision hidden dimension, enabling cross-modal transformation.
    """
    def __init__(self, config: Qwen2_5_VLConfig):
        # Initialize the parent class (Gemma3ForConditionalGeneration)
        super().__init__(config)
        self.vision_tower = UniQwenVisionTransformer(config.vision_config)
        self.spatial_scale = self.config.vision_config.spatial_merge_size **2
        self.flatten_size = config.vision_config.hidden_size * self.spatial_scale

        self.image_seq_len = 256
        self.max_img_cnt = 16
        self.max_prefix_len = self.image_seq_len * self.max_img_cnt

        # Add a regression head
        self.regression_head = RegressionHeadPerceiver(
                max_prefix_len=self.max_prefix_len * self.spatial_scale,
                text_hidden_size=config.vision_config.out_hidden_size,
                vision_hidden_size=config.vision_config.hidden_size, #self.flatten_size,
                num_heads=config.vision_config.num_heads, # * self.spatial_scale,
                cross_depth=2,
                self_septh=8)
            
        generation_type = os.environ.get("GENERATION_TYPE", "text_only")
        self.generation_type = generation_type  
        self.image_token_index = config.image_token_id
       
        self.post_init()


    # ------------------------------------------------------------------
    # weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module):
        """Initialize module weights following Transformer convention."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def concat_image_tokens(
        self,
        input_ids: torch.Tensor,
        boi_id: int,
        image_id: int,
        eoi_id: int,
        image_seq_len: int,
    ):
        """
        Concatenate [boi_id] + [image_id]*image_seq_len + [eoi_id] to the end of input_ids.
        Works for input_ids shaped as [S] or [B, S].
        Returns a tensor with the same dtype/device as input_ids, ready for model input.
        """
        assert input_ids.dtype in (torch.long, torch.int64, torch.int32), \
            f"input_ids dtype should normally be long; got {input_ids.dtype}"

        device = input_ids.device
        dtype = input_ids.dtype

        if input_ids.dim() == 1:
            # Shape [S]
            tail = torch.tensor(
                [boi_id] + [image_id]*image_seq_len + [eoi_id],
                dtype=dtype, device=device
            )
            return torch.cat([input_ids, tail], dim=0)

        elif input_ids.dim() == 2:
            # Shape [B, S]
            B = input_ids.size(0)
            boi = torch.full((B, 1), boi_id, dtype=dtype, device=device)
            imgs = torch.full((B, image_seq_len), image_id, dtype=dtype, device=device)
            eoi = torch.full((B, 1), eoi_id, dtype=dtype, device=device)
            return torch.cat([input_ids, boi, imgs, eoi], dim=1)

        else:
            raise ValueError(f"Unsupported input_ids shape: {input_ids.shape}")


    def get_vit_features(self, pixel_values, image_grid_thw, normalize=True):
        vision_outputs = self.vision_tower(pixel_values, image_grid_thw)
        
        if pixel_values.ndim == 2:
            vision_outputs = vision_outputs.reshape(-1, self.config.vision_config.hidden_size)
        elif pixel_values.ndim == 3:
            batch_size = pixel_values.shape[0]
            vision_outputs = vision_outputs.reshape(batch_size, -1, self.config.vision_config.hidden_size)
        return vision_outputs

    def fill_image_embeds(self, input_ids, pixel_values, image_grid_thw):
        """
        Fills the image token embeddings in the input_ids with the pixel values.
        This is a placeholder function and should be implemented based on the specific requirements.

        Args:
            input_ids (torch.LongTensor): The input IDs for the model.
            pixel_values (torch.FloatTensor): The pixel values of the images.

        Returns:
            torch.FloatTensor: The modified input IDs with image token embeddings filled.
        """
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds
    
    def get_context_mask(self, input_length, total_len=None):
        bs = input_length.shape[0]
        positions = torch.arange(total_len).unsqueeze(0).expand(bs, total_len).to(input_length.device)  # [bs, total_len]  
        cutoffs = (total_len - input_length).unsqueeze(1)  # [bs, 1]
        return positions >= cutoffs

    def perceiver_forward(self, input_embeds, shifted_outputs, input_mask, label_length):
        # input_embeds: (batch, seq_len, text_hidden_size)
        # hidden_states: (batch, seq_len, vision_hidden_size)
        bs = label_length.shape[0]
        input_length = input_mask.sum(dim=1)

        batch_context = torch.ones(bs, self.max_prefix_len, shifted_outputs.shape[-1], device=shifted_outputs.device).to(input_embeds.dtype)
        context_mask = self.get_context_mask(input_length, self.max_prefix_len)
        batch_context[context_mask] = input_embeds[input_mask]
        M_max = int(label_length.max().item())
        ## TODO: only applicable to training, since label padding is not implemented in Perceiver
        batch_labels = torch.ones(bs, M_max, shifted_outputs.shape[-1], device=shifted_outputs.device).to(shifted_outputs.dtype)
        ## TODO: only supports single label image
        out_mask = self.get_context_mask(label_length, total_len=self.image_seq_len)
        #batch_labels.masked_scatter(out_mask, shifted_outputs)
        batch_labels[out_mask] = shifted_outputs
        x = torch.cat([batch_context, batch_labels], dim=1)  # (batch, seq_len + image_seq_len, text_hidden_size)
        total_mask = torch.cat([context_mask, out_mask], dim=1)  # (batch, seq_len + image_seq_len)
        transformed_features = self.regression_head(x, context_mask, total_mask)  # (batch, image_seq_len, vision_hidden_size)
        vit_features = None
        if transformed_features.shape[-1] == self.flatten_size:
            transformed_features = transformed_features.contiguous().view(-1, self.config.vision_config.hidden_size)  # (batch * image_seq_len * (spatial_merge_size**2), vision_hidden_size)
            normed_vision_outputs = self.visual.merger.ln_q(transformed_features)
            vit_features = normed_vision_outputs.clone().view(bs, self.image_seq_len * self.spatial_scale, self.config.vision_config.hidden_size)
            transformed_features = normed_vision_outputs.view(bs, self.image_seq_len, self.flatten_size) # (batch, image_seq_len, vision_hidden_size * (spatial_merge_size**2))
            transformed_features = self.visual.merger.mlp(transformed_features)
        elif transformed_features.shape[-1] == self.config.vision_config.hidden_size:
            normed_vision_outputs = self.visual.merger.ln_q(transformed_features)
            vit_features = normed_vision_outputs.clone()
            transformed_features = normed_vision_outputs.view(bs, self.image_seq_len, self.flatten_size) # (batch, image_seq_len, vision_hidden_size * (spatial_merge_size**2))
            transformed_features = self.visual.merger.mlp(transformed_features)

        del input_embeds, batch_context, context_mask, total_mask, batch_labels

        return transformed_features.type_as(shifted_outputs), vit_features.type_as(shifted_outputs)
    
    def perceiver_inference(self, input_embeds, hidden_states, input_ids):
        prefix_image_mask = (input_ids == self.image_token_index).to(input_ids.device)
        eoi_mask = (input_ids == self.config.vision_end_token_id).to(input_ids.device)
        pos = torch.arange(input_ids.shape[-1], device=input_ids.device).unsqueeze(0) 
        last_eoi_pos = (eoi_mask * pos).max(dim=1).values  
        has_eoi = eoi_mask.any(dim=1)
        keep_mask = (pos < last_eoi_pos.unsqueeze(1)) | (~has_eoi.unsqueeze(1))
        prefix_image_mask &= keep_mask
        #prepare perceiver inputs
        bs = input_embeds.shape[0]
        batch_context = torch.ones(bs, self.max_prefix_len, hidden_states.shape[-1], device=hidden_states.device).to(input_embeds.dtype)
        input_length = prefix_image_mask.sum(dim=1)
        context_mask = self.get_context_mask(input_length, self.max_prefix_len)
        batch_context[context_mask] = input_embeds[prefix_image_mask]
        x = torch.cat([batch_context, hidden_states], dim=1)  # (batch, seq_len + image_seq_len, text_hidden_size)
        total_mask = torch.cat([context_mask, torch.ones(hidden_states.shape[:2], dtype=context_mask.dtype).to(context_mask.device)], dim=1)  # (batch, seq_len + image_seq_len)
        transformed_features = self.regression_head(x, context_mask, total_mask)  # (batch, image_seq_len, vision_hidden_size)
        vit_features = None
        if transformed_features.shape[-1] == self.flatten_size:
            transformed_features = transformed_features.contiguous().view(-1, self.config.vision_config.hidden_size)  # (batch * image_seq_len, vision_hidden_size)
            normed_vision_outputs = self.visual.merger.ln_q(transformed_features)
            vit_features = normed_vision_outputs.clone().view(bs, hidden_states.shape[1] * self.spatial_scale, self.config.vision_config.hidden_size)
            transformed_features = normed_vision_outputs.view(bs, hidden_states.shape[1], self.flatten_size)
            transformed_features = self.visual.merger.mlp(transformed_features)
        elif transformed_features.shape[-1] == self.config.vision_config.hidden_size:
            normed_vision_outputs = self.visual.merger.ln_q(transformed_features)
            vit_features = normed_vision_outputs.clone()
            transformed_features = normed_vision_outputs.view(bs, hidden_states.shape[1], self.flatten_size) # (batch, image_seq_len, vision_hidden_size * (spatial_merge_size**2))
            transformed_features = self.visual.merger.mlp(transformed_features)

        return transformed_features.type_as(input_embeds), vit_features.type_as(input_embeds)

    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        dtype = kwargs.get("torch_dtype", torch.bfloat16)

        # ===== Case A: First-time initialization from the official base model =====
        need_clone = "Qwen2.5-VL-7B-Instruct" in str(pretrained_model_name_or_path)

        if need_clone:
            print(f"[Init] Loading ORIGINAL base to clone from: {pretrained_model_name_or_path}")
            # Load the official base model to obtain its original state_dict
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
            base_sd = base_model.state_dict()
            config = base_model.config

            # Safety check: If the checkpoint already contains new module prefixes
            # (e.g., 'vision_tower.' or 'multi_modal_projector.'), it means
            # this is not the very first initialization — skip cloning.
            has_new_prefix = any(k.startswith("vision_tower.") for k in base_sd.keys()) or \
                             any(k.startswith("multi_modal_projector.") for k in base_sd.keys())  # change if your new MLP prefix is different
            if has_new_prefix:
                print("[Skip] Detected new module prefixes in checkpoint -> loading normally without cloning.")
                del base_model
                model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
                print(f"✅ Model loaded. dtype: {next(model.parameters()).dtype}")
                return model

            # ---- Cloning logic for first-time initialization ----
            new_sd = OrderedDict(base_sd)  # start from the original weights

            # (1) Copy model.visual.* -> vision_tower.* (exclude the whole visual.merger.* subtree)
            old_vis_prefix = "model.visual."
            new_vis_prefix = "vision_tower."
            exclude_prefix = "model.visual.merger.mlp"
            copied_vt = 0
            for k, v in base_sd.items():
                if k.startswith(old_vis_prefix) and not k.startswith(exclude_prefix):
                    vt_key = new_vis_prefix + k[len(old_vis_prefix):]
                    new_sd[vt_key] = v
                    copied_vt += 1

            print(f"[Clone] vision_tower.* params: {copied_vt}")

            # Initialize the new model (with vision_tower / new_mlp) and load the cloned state_dict
            del base_model
            model = cls(config).to(dtype)
            load_result = model.load_state_dict(new_sd, strict=False)

            # Report any missing/unexpected keys for debugging
            if load_result.missing_keys:
                print(f"!!! Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"!!! Unexpected keys (ignored): {load_result.unexpected_keys}")

        else:
            # ===== Case B: Fine-tuned model or any non-base checkpoint -> load normally =====
            print(f"[Normal] Loading model from '{pretrained_model_name_or_path}' ...")
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        print(f"✅ Model successfully loaded. Final model dtype: {next(model.parameters()).dtype}")
        return model


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
        ) -> Union[Tuple, UniQwenOutputWithImageLoss]:
        """
        Forward pass of the model that extends the original Gemma3ForConditionalGeneration's forward method.
        This implementation adds feature transformation and image loss computation.

        Args:
            *args: Variable length argument list passed to the original forward method.
            **kwargs: Arbitrary keyword arguments passed to the original forward method.

        Returns:
            outputs (GemmaGenOutputWithImageLoss): Original model outputs plus:
                - transformed_features (torch.Tensor): Features transformed by the regression head
                - image_loss (torch.Tensor): Loss for image feature regression, if computed
        """
        # Get outputs from the parent model's forward pass
        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask, 
                                  position_ids=position_ids, 
                                  past_key_values=past_key_values, 
                                  inputs_embeds=inputs_embeds,
                                  labels=labels, 
                                  use_cache=use_cache, 
                                  output_attentions=output_attentions, 
                                  output_hidden_states=output_hidden_states, 
                                  pixel_values=pixel_values, 
                                  pixel_values_videos=pixel_values_videos,
                                  image_grid_thw=image_grid_thw,
                                  video_grid_thw=video_grid_thw,
                                  rope_deltas=rope_deltas,
                                  second_per_grid_ts=second_per_grid_ts,
                                  cache_position=cache_position,
                                  logits_to_keep=logits_to_keep,
                                  **kwargs 
                                  )

        image_mask = (labels == self.image_token_index).to(labels.device) if labels is not None else None  # (B, seq_len) bool tensor
        image_loss = None
        if image_mask is not None and torch.any(image_mask):            
            input_image_mask = (input_ids == self.image_token_index).to(input_ids.device).logical_and(~image_mask)
            if inputs_embeds == None:
                inputs_embeds = self.fill_image_embeds(input_ids, pixel_values, image_grid_thw)
            prompt_img_cnt = input_image_mask.sum(dim=1) // self.image_seq_len
            label_img_cnt = image_mask.sum(dim=1) // self.image_seq_len
            image_lengths = torch.stack([prompt_img_cnt, label_img_cnt], dim=1).reshape(-1)
            segment_ids      = torch.arange(image_lengths.size(0), device=image_lengths.device)
            seg_masks = torch.repeat_interleave(segment_ids % 2 == 1, image_lengths)
            label_pixel_values = pixel_values.reshape(-1, self.image_seq_len * self.spatial_scale, pixel_values.size(-1))[seg_masks]
            label_image_grid_thw = image_grid_thw[seg_masks]
            #print(f'label pixel values shape: {label_pixel_values.shape}')
            target_vit = self.get_vit_features(label_pixel_values.reshape(-1, pixel_values.size(-1)), label_image_grid_thw)
            target_features = self.visual(label_pixel_values.reshape(-1, pixel_values.size(-1)), grid_thw=label_image_grid_thw)
            shifted_outputs  = outputs.hidden_states[-1][:, :-1, :][image_mask[:, 1:]].clone()  # [B, L‑1, H] Every location where the next token (t + 1) is an image token

            # 3) projection → loss
            transformed_features, transformed_vit = self.perceiver_forward(inputs_embeds, shifted_outputs, input_image_mask, image_mask.sum(dim=1))
            loss_type = kwargs.pop("loss_type", "mse")
            transformed_vit = transformed_vit.reshape(-1, target_vit.shape[-1])  # [N_img_tokens, vision_hidden_size]
            transformed_features = transformed_features.reshape(-1, target_features.shape[-1])  # [N_img_tokens, vision_hidden_size * (spatial_merge_size**2)]
            if loss_type == 'mse':
                image_loss = nn.functional.mse_loss(
                    transformed_vit,
                    target_vit,
                )
            elif loss_type == 'l1':
                image_loss = nn.functional.l1_loss(
                    transformed_vit,
                    target_vit,
                ) 
                image_loss += nn.functional.l1_loss(
                    transformed_features,
                    target_features,
                )
            elif loss_type == 'cosine':
                cos_sim_vit = nn.functional.cosine_similarity(
                    transformed_vit,
                    target_vit,
                    dim=-1,
                )
                cos_sim_features += nn.functional.cosine_similarity(
                    transformed_features,
                    target_features,
                    dim=-1,
                )
                image_loss = 0.5 * (1.0 - cos_sim_vit).mean() + 0.5 * (1.0 - cos_sim_features).mean()
            del transformed_features, transformed_vit, target_vit
           
        return UniQwenOutputWithImageLoss(
            image_loss=image_loss,
            loss=image_loss,   # loss
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: None,
        **model_kwargs,
        ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        boi_id = self.config.vision_start_token_id
        eoi_id = self.config.vision_end_token_id
        image_id = self.image_token_index
        action_id = 522
        fill_idx = -1
        filling_path = os.getenv("FILLING_PATH", None)
        img_emb_path = os.getenv("IMG_EMB_PATH", -1)
        wait_for_filling = False 

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )
        

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            # If we use FA2 and a static cache, we cannot compile with fullgraph
            if self.config._attn_implementation == "flash_attention_2" and getattr(
                model_kwargs.get("past_key_values"), "is_compileable", False
            ):
                if generation_config.compile_config is None:
                    generation_config.compile_config = CompileConfig(fullgraph=False)
                # only raise warning if the user passed an explicit compile-config (otherwise, simply change the default without confusing the user)
                elif generation_config.compile_config.fullgraph:
                    logger.warning_once(
                        "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
                        "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
                    )
                    generation_config.compile_config.fullgraph = False
            model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        image_grid_thw = model_kwargs["image_grid_thw"] 
        
        inputs_embeds = self.fill_image_embeds(input_ids, model_kwargs.pop("pixel_values", None), image_grid_thw)
        ## Generated image must have the same shape as the input image
        inputs_vit_feats = torch.randn(batch_size, inputs_embeds.shape[1], self.spatial_scale, self.config.vision_config.hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        in_image = input_ids.new_zeros(batch_size, dtype=torch.bool)
        in_action = input_ids.new_zeros(batch_size, dtype=torch.long)
        output_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        next_token_emb = None
        #hidden states of generated unfinished image tokens
        image_hidden_states = torch.randn(batch_size, self.image_seq_len, self.config.vision_config.out_hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        vision_positions = None

        # is_prefill = True
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": False})
            model_inputs.update({"output_hidden_states": True})
            model_inputs.update({"inputs_embeds": next_token_emb} if next_token_emb is not None else {"inputs_embeds": inputs_embeds})
            # _ = model_inputs.pop("input_ids", None)  # remove input_ids from model_inputs to avoid passing it to the model
            # _ = model_inputs.pop("pixel_values", None)  # remove pixel_values from model_inputs to avoid passing it to the model

            # update position_ids
            if vision_positions is not None:
                model_inputs["position_ids"][1:] =  vision_positions[:, :, len(input_ids[-1]) - 1].unsqueeze(-1) 
            
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            token_emb = self.get_input_embeddings()(next_tokens)  # (B, hidden)
            token_vit_feats = torch.randn(batch_size, self.spatial_scale, self.config.vision_config.hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            
            in_action[in_action > 0] += 1
            if (in_action==4).any() and self.generation_type == 'multimodal':
                next_tokens[in_action==4] = boi_id
                token_emb[in_action==4] = self.get_input_embeddings()(next_tokens[in_action==4])  
                in_action[in_action==4] = 0  
            elif self.generation_type == 'agent':
                if wait_for_filling:
                    filling_path = wait_for_env("FILLING_PATH")
                    if filling_path == "END":
                        return input_ids, None, None
                    elif os.path.exists(filling_path):
                        print(f"filling from {filling_path}")
                        time.sleep(2) 
                        fill_text_ids = torch.load(filling_path).reshape(-1).to(input_ids.device)
                        fill_len = fill_text_ids.shape[0]
                        fill_idx = 0
                        wait_for_filling = False
                if fill_idx == -1:
                    next_tokens.fill_(boi_id)
                    token_emb = self.get_input_embeddings()(next_tokens)
                elif fill_idx < fill_len:
                    next_tokens.fill_(fill_text_ids[fill_idx].item())
                    token_emb = self.get_input_embeddings()(next_tokens)
                    fill_idx += 1
                else:
                    fill_idx = -1

            # in the image token sequence
            if in_image.any():
                img_seq_len = (input_ids == self.image_token_index).logical_and(output_mask).sum(dim=1)
                boi_cnt = (input_ids == boi_id).logical_and(output_mask).sum(dim=1)
                #fin_image = (img_seq_len > 0).logical_and(img_seq_len % self.image_seq_len == 0)
                fin_image = (img_seq_len // self.image_seq_len) == boi_cnt
                if fin_image.any():
                    next_tokens[fin_image] = eoi_id
                    token_emb[fin_image] = self.get_input_embeddings()(next_tokens[fin_image])  # (B, hidden)
                in_image &= ~fin_image
                if in_image.any():
                    next_tokens[in_image] = image_id
                    # compute projection for all, but mask non-image samples
                    last_hidden = outputs.hidden_states[-1][:, -1, :]  # (B, hidden)
                    batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
                    #fill last hidden to unfinished image tokens
                    image_hidden_states[batch_idx, img_seq_len % self.image_seq_len, :] = last_hidden
                    proj, vit_feat = self.perceiver_inference(inputs_embeds, image_hidden_states, input_ids)
                    proj = proj[batch_idx, img_seq_len % self.image_seq_len, :]
                    vit_feat = vit_feat[batch_idx, (img_seq_len % self.image_seq_len) * self.spatial_scale : (img_seq_len % self.image_seq_len + 1) * self.spatial_scale, :] # (B, spatial_scale, hidden)
                    token_emb[in_image] = proj[in_image]           # (B, hidden)
                    if vit_feat is not None:
                        token_vit_feats[in_image] = vit_feat[in_image]

            # per-sample BOI/EOI detection
            is_boi = next_tokens == boi_id
            is_eoi = next_tokens == eoi_id
            if is_boi.any():
                print(f"boi detected")
                # prefill image tokens to get the correct rope index: [input_ids, boi, image..., eoi ]
                prefilling_img_input_ids = self.concat_image_tokens(input_ids, boi_id, image_id, eoi_id, self.image_seq_len)
                next_image_grid_thw = image_grid_thw[-1].unsqueeze(0)
                image_grid_thw = torch.cat([image_grid_thw, next_image_grid_thw], dim=0)
                # get rope index for the new image
                vision_positions, rope_deltas = self.model.get_rope_index(
                    prefilling_img_input_ids,
                    image_grid_thw=image_grid_thw
                ) 
                
                
            #in_image &= ~is_eoi 
            in_image |= is_boi
            if is_eoi.any():
                print(f"eoi detected, {(input_ids == self.image_token_index).logical_and(output_mask).sum()} image tokens")
                self.model.rope_deltas = rope_deltas
                vision_positions = None

            if (next_tokens == action_id).any():
                in_action[next_tokens == action_id] += 1

            # update input_ids and cache
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            inputs_embeds = torch.cat([inputs_embeds, token_emb.unsqueeze(1)], dim=1)
            output_mask = torch.cat([output_mask, torch.ones(batch_size, 1, dtype=torch.bool).to(output_mask.device)], dim=1)
            next_token_emb = token_emb.unsqueeze(1)
            inputs_vit_feats = torch.cat([inputs_vit_feats, token_vit_feats.unsqueeze(1)], dim=1)

            if is_eoi.any() and self.generation_type == 'agent':
                wait_for_filling = True
                os.environ["FILLING_PATH"] = ""
                img_seq_len = (input_ids == self.image_token_index).logical_and(output_mask).sum(dim=1)
                valid_img_cnt = img_seq_len // self.image_seq_len
                valid_img_seq_len = valid_img_cnt * self.image_seq_len
                image_mask = (input_ids == self.image_token_index).logical_and(output_mask)
                image_vit_feats = inputs_vit_feats[image_mask][:valid_img_seq_len].view(-1, inputs_vit_feats.shape[-1])
                image_vit_feats = image_vit_feats.unsqueeze(0).view(-1, self.image_seq_len * self.spatial_scale, image_vit_feats.shape[-1])
                sample_idx = os.getenv("SAMPLE_IDX", None)
                round_idx = str((valid_img_cnt-1).item())
                save_path = os.path.join(img_emb_path, f"img_embeds_{sample_idx}_{round_idx}.pt")
                torch.save(image_vit_feats[-1].cpu(), save_path)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        print(f'generated id length:{input_ids.shape}')
        img_seq_len = (input_ids == self.image_token_index).logical_and(output_mask).sum(dim=1)
        boi_cnt = (input_ids == boi_id).logical_and(output_mask).sum(dim=1)
        print('generated image cnt:', boi_cnt)
        valid_img_cnt = img_seq_len // self.image_seq_len
        print('valid image cnt:', valid_img_cnt)
        valid_img_seq_len = valid_img_cnt * self.image_seq_len
         

        image_mask = (input_ids == self.image_token_index).logical_and(output_mask)
        image_embeds = inputs_embeds[image_mask][:valid_img_seq_len].view(-1, inputs_embeds.shape[-1])  # [batch_size * num_img_tokens, hidden_size]
        image_embeds = image_embeds.view(-1, self.image_seq_len, image_embeds.shape[-1])

        image_vit_feats = inputs_vit_feats[image_mask][:valid_img_seq_len].view(-1, inputs_vit_feats.shape[-1])
        image_vit_feats = image_vit_feats.unsqueeze(0).view(-1, self.image_seq_len * self.spatial_scale, image_vit_feats.shape[-1])
        print(f'generated image length:{image_embeds.shape[0]}')

        return input_ids, image_embeds, image_vit_feats