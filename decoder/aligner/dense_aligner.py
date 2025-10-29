import torch
import torch.nn.functional as F
from torch import nn
from torchscale.architecture.encoder import Encoder
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.config import EncoderDecoderConfig
from torchscale.component.embedding import PositionalEmbedding
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
import os, sys
import open_clip

class ClipToLatentAligner(nn.Module):
    def __init__(self, vae_encoder, input_dim=1024, out_dim=512, grid_size = 28, layer = 12, causal_mask=False):
        super().__init__()

        self.out_dim = out_dim
        self.grid_size = grid_size
        cfg = EncoderDecoderConfig(
            checkpoint_activations=True,
            flash_attention=True,
            encoder_embed_dim=out_dim,
            decoder_embed_dim=out_dim,
            encoder_attention_heads=8,
            decoder_attention_heads=8,
            encoder_layers=layer,
            decoder_layers=layer,
        )
        self.encoder_proj = Encoder(
            cfg,
            embed_tokens=nn.Linear(input_dim, out_dim),
            embed_positions=PositionalEmbedding(32768, out_dim),
            is_encoder_decoder=True,
        )
        self.encoder_query = nn.Parameter(torch.randn(grid_size * grid_size, out_dim))
        self.encoder = Decoder(
            cfg,
            is_encoder_decoder=True,
            causal_mask=causal_mask,
        )

        """ self.conv_norm_out = nn.GroupNorm(num_channels=512, num_groups=32, eps=1e-6)
        self.conv_act = nn.SiLU() """
        self.conv_out = nn.Conv2d(out_dim, 8, 3, padding=1)
        self.quant_conv = nn.Conv2d(8, 8, 1)
        # Load the weights from the pretrained model
        self.load_pretrained_weights(vae_encoder)
        
        # Freeze the weights of conv_out and quant_conv
        self.freeze_layers()

    def load_pretrained_weights(self, pretrained_model):
        if pretrained_model is None:
            return
        # Assuming pretrained_model is a model from which we load the weights
        self.conv_out.weight.data = pretrained_model.conv_out.weight.data.clone()
        self.conv_out.bias.data = pretrained_model.conv_out.bias.data.clone()
        self.quant_conv.weight.data = pretrained_model.quant_conv.weight.data.clone()
        self.quant_conv.bias.data = pretrained_model.quant_conv.bias.data.clone()

    def freeze_layers(self):
        # Freeze conv_out and quant_conv layers
        for param in self.conv_out.parameters():
            param.requires_grad = False

        for param in self.quant_conv.parameters():
            param.requires_grad = False


    def forward(self, condition, padding_mask, vae_embed):
        gpt_embed, output = self.encode(condition, padding_mask)
        mse_loss = F.mse_loss(gpt_embed.float(), vae_embed.float(), reduction='mean')

        return {'mse_loss': mse_loss, 'output':output}

    def encode(self, condition, padding_mask):
        condition = self.encoder_proj(
            src_tokens=condition,
            encoder_padding_mask=padding_mask,
        )
        condition = self.encoder(
            prev_output_tokens=None,
            token_embeddings=self.encoder_query.unsqueeze(0).expand(condition["encoder_out"].shape[1], -1, -1),
            encoder_out=condition,
        )[0]
        bs = condition.size(0)
        output = condition.transpose(1, 2).contiguous().view(bs, self.out_dim, self.grid_size, self.grid_size)
        output = self.conv_out(output)
        output = self.quant_conv(output)
        posterior = DiagonalGaussianDistribution(output)
        return condition, AutoencoderKLOutput(latent_dist=posterior)