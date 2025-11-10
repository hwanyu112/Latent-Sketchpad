import torch
import torch.nn as nn
from transformers import Gemma3ForConditionalGeneration, Gemma3Config, GenerationConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3CausalLMOutputWithPast
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Union, Tuple, List
from transformers.cache_utils import Cache
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
import torch.nn.functional as F
from transformers.generation.utils import GenerateOutput
import os
import sys
sys.append = os.path.dirname(os.path.abspath(__file__))
from model.perceiver import PerceiverAR

class RegressionHeadWithAttention(nn.Module):
    def __init__(self, text_hidden_size, vision_hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        # multi‐head self‐attention
        self.attn = nn.MultiheadAttention(embed_dim=text_hidden_size,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)  # batch_first makes life easier
        # a residual + layer‐norm around the attention
        self.norm1 = nn.LayerNorm(text_hidden_size)
        # final projection MLP
        self.mlp = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, vision_hidden_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, text_hidden_size)
        # if your "text" is a single vector per sample, wrap it as seq_len=1:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, hidden)

        #assert not torch.isnan(x).any()

        attn_out, _ = self.attn(x, x, x)
        #assert not torch.isnan(attn_out).any()

        x_res = x + attn_out
        #assert not torch.isnan(x_res).any()

        x = self.norm1(x_res)
        #assert not torch.isnan(x).any()

        out = self.mlp(x.squeeze(1))
        #assert not torch.isnan(out).any()
        return out

class RegressionHeadPerceiver(nn.Module):
    def __init__(self, text_hidden_size, vision_hidden_size, max_prefix_len = 8192, num_heads=8, dropout=0.1):
        super().__init__()
        self.perceiver = PerceiverAR(
            dim = vision_hidden_size, 
            depth = 8, 
            dim_head = vision_hidden_size // num_heads, 
            heads = num_heads, 
            max_seq_len = 32768, 
            cross_attn_seq_len = max_prefix_len, 
            cross_attn_dropout = dropout,
            perceive_depth = 2,
        )
        self.mlp = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, vision_hidden_size),
        )
        self.dim = vision_hidden_size

    def forward(self, x, prefix_mask=None, total_mask=None):
        # x: (batch, seq_len, text_hidden_size)
        out = torch.randn(x.shape[0], x.shape[1], self.dim, device=x.device, dtype=x.dtype)
        out[total_mask] = self.mlp(x[total_mask])  # (batch * seq_len, vision_hidden_size)
        assert not torch.isnan(out).any()
        out = self.perceiver(out, prefix_mask=prefix_mask)
        assert not torch.isnan(out).any()
        return out

@dataclass
class GemmaGenOutputWithImageLoss(Gemma3CausalLMOutputWithPast):
    """
    Extends Gemma3CausalLMOutputWithPast to include image loss and transformed features.

    Args:
        image_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss computed for the image feature regression task.
        transformed_features (`torch.FloatTensor` of shape `(batch_size, vision_hidden_size)`, *optional*):
            Features transformed by the regression head.
    """
    image_loss: Optional[torch.FloatTensor] = None
    transformed_features: Optional[torch.FloatTensor] = None
    
class GemmaGenForConditionalGeneration(Gemma3ForConditionalGeneration):
    """
    Extended version of Gemma3ForConditionalGeneration with an additional regression head.
    This class inherits from Gemma3ForConditionalGeneration and adds a linear regression layer
    for additional processing capabilities.

    Args:
        config (Gemma3Config): The configuration object containing model specifications.
                              Must include text_config.hidden_size and vision_config.hidden_size.
    
    Attributes:
        regression_head (nn.Linear): A linear layer that projects features from text hidden dimension
                                   to vision hidden dimension, enabling cross-modal transformation.
    """
    def __init__(self, config: Gemma3Config):
        # Initialize the parent class (Gemma3ForConditionalGeneration)
        super().__init__(config)

        # Add a regression head
        self.regression_head = RegressionHeadPerceiver(
                text_hidden_size=config.text_config.hidden_size,
                vision_hidden_size=config.vision_config.hidden_size,
                num_heads=config.vision_config.num_attention_heads)
            
        generation_type = os.environ.get("GENERATION_TYPE", "text_only")
        self.generation_type = generation_type  
        # token index of the image token
        self.image_token_index = config.image_token_index
        self.image_seq_len = 256
        self.max_img_cnt = 32
        self.max_prefix_len = self.image_seq_len * self.max_img_cnt
        self.patches_per_image = 64
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

    def fill_image_embeds(self, input_ids, pixel_values):
        """
        Fills the image token embeddings in the input_ids with the pixel values.
        This is a placeholder function and should be implemented based on the specific requirements.

        Args:
            input_ids (torch.LongTensor): The input IDs for the model.
            pixel_values (torch.FloatTensor): The pixel values of the images.

        Returns:
            torch.FloatTensor: The modified input IDs with image token embeddings filled.
        """
        image_features = self.get_image_features(pixel_values.squeeze(0) if pixel_values.ndim > 4 else pixel_values)
        inputs_embeds = self.get_input_embeddings()(input_ids)
        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
        return inputs_embeds
    
    def vision_forward(self, last_hidden):
        transformed_features = self.regression_head(last_hidden)    # [N_img_tokens, H] or [N_img_tokens, vision_hidden_size]
        vit_features = None
        if transformed_features.shape[-1] == self.config.vision_config.hidden_size:
            normed_vision_outputs = self.multi_modal_projector.mm_soft_emb_norm(transformed_features)
            vit_features = transformed_features.clone()
            transformed_features = torch.matmul(normed_vision_outputs, self.multi_modal_projector.mm_input_projection_weight)
        return transformed_features.type_as(last_hidden), vit_features.type_as(last_hidden)

    def get_vit_features(self, pixel_values):
        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        batch_size, _, hidden_dim = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, hidden_dim, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.multi_modal_projector.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)
        normed_vision_outputs = self.multi_modal_projector.mm_soft_emb_norm(pooled_vision_outputs)
        return normed_vision_outputs.type_as(vision_outputs)
    
    def get_context_mask(self, input_length, total_len=None):
        bs = input_length.shape[0]
        positions = torch.arange(total_len).unsqueeze(0).expand(bs, total_len).to(input_length.device)  # [bs, total_len]  
        cutoffs = (total_len - input_length).unsqueeze(1)  # [bs, 1]
        return positions >= cutoffs

    def perceiver_forward(self, input_embeds, shifted_outputs, input_mask, label_length):
        # input_embeds: (batch, seq_len, text_hidden_size)
        # hidden_states: (batch, seq_len, vision_hidden_size)
        bs = input_embeds.shape[0]
        input_length = input_mask.sum(dim=1)

        batch_context = torch.ones(bs, self.max_prefix_len, shifted_outputs.shape[-1], device=shifted_outputs.device).to(input_embeds.dtype)
        context_mask = self.get_context_mask(input_length, self.max_prefix_len)
        batch_context[context_mask] = input_embeds[input_mask]
        M_max = int(label_length.max().item())
        ## TODO: only applicable to training, since label padding is not implemented in Perceiver
        batch_labels = torch.ones(bs, M_max, shifted_outputs.shape[-1], device=shifted_outputs.device).to(shifted_outputs.dtype)
        out_mask = self.get_context_mask(label_length, total_len=self.image_seq_len)
        #batch_labels.masked_scatter(out_mask, shifted_outputs)
        batch_labels[out_mask] = shifted_outputs
        x = torch.cat([batch_context, batch_labels], dim=1)  # (batch, seq_len + image_seq_len, text_hidden_size)
        total_mask = torch.cat([context_mask, out_mask], dim=1)  # (batch, seq_len + image_seq_len)
        transformed_features = self.regression_head(x, context_mask, total_mask)  # (batch, image_seq_len, vision_hidden_size)
        vit_features = None
        if transformed_features.shape[-1] == self.config.vision_config.hidden_size:
            transformed_features = transformed_features.view(-1, self.config.vision_config.hidden_size)  # (batch * image_seq_len, vision_hidden_size)
            normed_vision_outputs = self.multi_modal_projector.mm_soft_emb_norm(transformed_features)
            vit_features = normed_vision_outputs.clone().view(bs, self.image_seq_len, self.config.vision_config.hidden_size)
            transformed_features = torch.matmul(normed_vision_outputs, self.multi_modal_projector.mm_input_projection_weight)
        return transformed_features.type_as(shifted_outputs), vit_features.type_as(input_embeds)
    
    def perceiver_inference(self, input_embeds, hidden_states, input_ids):
        prefix_image_mask = (input_ids == self.image_token_index).to(input_ids.device)
        eoi_mask = (input_ids == self.config.eoi_token_index).to(input_ids.device)
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
        if transformed_features.shape[-1] == self.config.vision_config.hidden_size:
            transformed_features = transformed_features.view(-1, self.config.vision_config.hidden_size)  # (batch * image_seq_len, vision_hidden_size)
            normed_vision_outputs = self.multi_modal_projector.mm_soft_emb_norm(transformed_features)
            vit_features = normed_vision_outputs.clone().view(bs, hidden_states.shape[1], self.config.vision_config.hidden_size)
            transformed_features = torch.matmul(normed_vision_outputs, self.multi_modal_projector.mm_input_projection_weight)
            transformed_features = transformed_features.view(bs, hidden_states.shape[1], self.config.text_config.hidden_size)
        return transformed_features.type_as(input_embeds), vit_features.type_as(input_embeds)


    def forward(self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0, 
        **kwargs):
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
                                  pixel_values=pixel_values, 
                                  attention_mask=attention_mask, 
                                  position_ids=position_ids, 
                                  past_key_values=past_key_values, 
                                  token_type_ids=token_type_ids, 
                                  cache_position=cache_position, 
                                  inputs_embeds=inputs_embeds,
                                  labels=labels, 
                                  use_cache=use_cache, 
                                  output_attentions=output_attentions, 
                                  output_hidden_states=output_hidden_states, 
                                  return_dict=return_dict, 
                                  logits_to_keep=logits_to_keep, **kwargs)

         # Get features at the image token position
        image_mask = (labels == self.image_token_index).to(labels.device) if labels is not None else None  # (B, seq_len) bool tensor
        custom_outputs = GemmaGenOutputWithImageLoss(**outputs)
        if image_mask is not None and torch.any(image_mask):            
            inputs_embeds = self.fill_image_embeds(input_ids, pixel_values)
            input_image_mask = (input_ids == self.image_token_index).to(input_ids.device).logical_and(~image_mask)
            # 2) SHIFT everything by one step (drop last token for targets, first for preds)
            #    ‑ shifted_* : tensors with shape [B, L‑1, H]
            target_features  = inputs_embeds[image_mask]
            shifted_outputs  = outputs.hidden_states[-1][:, :-1, :][image_mask[:, 1:]]  # [B, L‑1, H] Every location where the next token (t + 1) is an image token
            prompt_img_cnt = input_image_mask.sum(dim=1) // self.image_seq_len
            label_img_cnt = image_mask.sum(dim=1) // self.image_seq_len
            image_lengths = torch.stack([prompt_img_cnt, label_img_cnt], dim=1).reshape(-1)
            segment_ids      = torch.arange(image_lengths.size(0), device=image_lengths.device)
            seg_masks = torch.repeat_interleave(segment_ids % 2 == 1, image_lengths)
            label_pixel_values = pixel_values[seg_masks]
            #print(f'label pixel values shape: {label_pixel_values.shape}')
            target_vit = self.get_vit_features(label_pixel_values)

            # 3) projection → loss
            transformed_features, transformed_vit = self.perceiver_forward(inputs_embeds, shifted_outputs, input_image_mask, image_mask.sum(dim=1))
            loss_type = kwargs.pop("loss_type", "mse")
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
            elif loss_type == 'cosine':
                cos_sim = nn.functional.cosine_similarity(
                    transformed_vit,
                    target_vit,
                    dim=-1,
                )
                image_loss = 0.5 * (1.0 - cos_sim).mean()
            custom_outputs.image_loss = image_loss
        return custom_outputs

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

        boi_id = self.config.boi_token_index
        eoi_id = self.config.eoi_token_index
        image_id = self.image_token_index
        action_id = 954

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
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            if getattr(self, "hf_quantizer", None) is not None:
                is_compileable &= self.hf_quantizer.is_compileable
            is_compileable = is_compileable and not generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        if getattr(generation_config, "prefill_chunk_size", None) is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        inputs_embeds = self.fill_image_embeds(input_ids, model_kwargs.pop("pixel_values", None))
        inputs_vit_feats = torch.randn(batch_size, inputs_embeds.shape[1], self.config.vision_config.hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        in_image = input_ids.new_zeros(batch_size, dtype=torch.bool)
        in_action = input_ids.new_zeros(batch_size, dtype=torch.long)
        output_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        next_token_emb = None
        #hidden states of generated unfinished image tokens
        image_hidden_states = torch.randn(batch_size, self.image_seq_len, self.config.text_config.hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        next_token_type = None

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": True})
            model_inputs.update({"output_hidden_states": True})
            model_inputs.update({"inputs_embeds": next_token_emb} if next_token_emb is not None else {"inputs_embeds": inputs_embeds})
            if next_token_type is not None:
                model_inputs.update({"token_type_ids": next_token_type})
            _ = model_inputs.pop("input_ids", None)  # remove input_ids from model_inputs to avoid passing it to the model
            _ = model_inputs.pop("pixel_values", None)  # remove pixel_values from model_inputs to avoid passing it to the model

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
            token_vit_feats = torch.randn(batch_size, self.config.vision_config.hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)            
            
            in_action[in_action > 0] += 1
            if (in_action==4).any() and self.generation_type == 'multimodal':
                next_tokens[in_action==4] = boi_id
                token_emb[in_action==4] = self.get_input_embeddings()(next_tokens[in_action==4])  
                in_action[in_action==4] = 0  

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
                    vit_feat = vit_feat[batch_idx, img_seq_len % self.image_seq_len, :]
                    token_emb[in_image] = proj[in_image]           # (B, hidden)
                    if vit_feat is not None:
                        token_vit_feats[in_image] = vit_feat[in_image]

            # per-sample BOI/EOI detection
            is_boi = next_tokens == boi_id
            is_eoi = next_tokens == eoi_id
            if is_boi.any():
                print(f"boi detected")
            #in_image &= ~is_eoi 
            in_image |= is_boi
            if is_eoi.any():
                print(f"eoi detected, {(input_ids == self.image_token_index).logical_and(output_mask).sum()} image tokens")

            if (next_tokens == action_id).any():
                in_action[next_tokens == action_id] += 1

            # update input_ids and cache
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            inputs_embeds = torch.cat([inputs_embeds, token_emb.unsqueeze(1)], dim=1)
            output_mask = torch.cat([output_mask, torch.ones(batch_size, 1, dtype=torch.bool).to(output_mask.device)], dim=1)
            next_token_emb = token_emb.unsqueeze(1)
            if next_token_type is None:
                next_token_type = torch.zeros_like(next_tokens, dtype=next_tokens.dtype).unsqueeze(-1)
            next_token_type[next_tokens.unsqueeze(-1) == self.image_token_index] = 1
            inputs_vit_feats = torch.cat([inputs_vit_feats, token_vit_feats.unsqueeze(1)], dim=1)


            #cur_len += 1
            # finished sentences should have their next token be a padding token

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
        image_vit_feats = image_vit_feats.view(-1, self.image_seq_len, image_vit_feats.shape[-1])
        print(f'generated image length:{image_embeds.shape[0]}')

        return input_ids, image_embeds, image_vit_feats