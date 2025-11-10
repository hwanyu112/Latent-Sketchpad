import io
import os
import sys
import torch
import random
import numpy as np
from diffusers.models import AutoencoderKL
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from gradio import networking
import secrets
import argparse
import uvicorn
from copy import deepcopy

# adjust import paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'aligner'))
from aligner.dense_aligner import ClipToLatentAligner
sys.path.append(os.path.dirname(__file__))
from vision_encoder_wrapper import VisionTransformerWrapper

# ---------------------------
# Random seed & device setup
# ---------------------------
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------
# Model loading function
# ---------------------------
def load_models(vision_model, checkpoint_path, image_size, feature_dim, layers):
    vit_feature_only = 1  # if using feature extraction only
    vision_encoder = VisionTransformerWrapper(vision_model, checkpoint_path, image_size, vit_feature_only)
    vision_encoder.move_to(DEVICE)

    vae_ref = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(DEVICE)
    vae_ref.eval()

    grid_size = image_size // 8
    aligner_net = ClipToLatentAligner(None, feature_dim, 512, grid_size, layers).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = {k.replace('aligner_net.', ''): v for k, v in checkpoint['state_dict'].items()}
    aligner_net.load_state_dict(state_dict)
    aligner_net.eval()

    return vision_encoder, aligner_net, vae_ref


# ---------------------------
# Core reconstruction logic
# ---------------------------
def decode_latent(vae_ref, latent_tensor):
    decoded = vae_ref.decode(latent_tensor).sample
    tensor = (decoded.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(tensor.cpu())


def reconstruct_from_image(img, vision_encoder, aligner_net, vae_ref):
    if img is None:
        return None
    inp = vision_encoder.image_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        tokens = vision_encoder.encode_image(inp)
        mask = torch.zeros((tokens.size(0), tokens.size(1)), dtype=torch.bool).to(DEVICE)
        _, latent_data = aligner_net.encode(tokens, mask)
        latent = latent_data.latent_dist.mode()
        rec_img = decode_latent(vae_ref, latent)
    return rec_img


def reconstruct_from_tensor(tensor, aligner_net, vae_ref):
    if tensor is None:
        return None
    inp = tensor.to(DEVICE).to(torch.float32)
    if inp.ndim < 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        mask = torch.zeros(inp.shape[:2], dtype=torch.bool).to(DEVICE)
        _, latent_data = aligner_net.encode(inp, mask)
        latent = latent_data.latent_dist.mode()
        return decode_latent(vae_ref, latent)


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI()


@app.post('/reconstruct_image')
async def api_reconstruct_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid image file')

    out = reconstruct_from_image(img, app.state.vision_encoder, app.state.aligner_net, app.state.vae_ref)
    buf = io.BytesIO()
    out.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/png')


@app.post('/reconstruct_tensor')
async def api_reconstruct_tensor(file: UploadFile = File(...)):
    try:
        data = torch.load(io.BytesIO(await file.read()), map_location=DEVICE)
        tensor = data['tensor'] if isinstance(data, dict) and 'tensor' in data else data
        if not isinstance(tensor, torch.Tensor):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid tensor file')

    out = reconstruct_from_tensor(tensor, app.state.aligner_net, app.state.vae_ref)
    buf = io.BytesIO()
    out.save(buf, format='PNG')
    buf.seek(0)
    return StreamingResponse(buf, media_type='image/png')


# ---------------------------
# Gradio UI
# ---------------------------
def build_gradio_ui(vision_encoder, aligner_net, vae_ref):
    demo = gr.Blocks()
    with demo:
        gr.Markdown('# Align+VAE Reconstruction')

        with gr.Tab('Image Input'):
            inp_img = gr.Image(type='pil')
            out_img = gr.Image(type='pil')
            inp_img.change(
                fn=lambda img: reconstruct_from_image(img, vision_encoder, aligner_net, vae_ref),
                inputs=inp_img,
                outputs=out_img
            )

        with gr.Tab('Tensor Input (.pt)'):
            inp_file = gr.File(file_types=['.pt'])
            out_tensor_img = gr.Image(type='pil')

            def _gr_from_tensor(f):
                if f is None:
                    return None
                data = torch.load(f.name, map_location=DEVICE)
                tensor = data['tensor'] if isinstance(data, dict) and 'tensor' in data else data
                if not isinstance(tensor, torch.Tensor):
                    return None
                return reconstruct_from_tensor(tensor, aligner_net, vae_ref)

            inp_file.change(fn=_gr_from_tensor, inputs=inp_file, outputs=out_tensor_img)
    return demo


# ---------------------------
# Argument parser
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Latent Sketchpad Reconstruction Server")

    parser.add_argument(
        "--vision_model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Name or path of the vision backbone model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the pretrained checkpoint."
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1280,
        help="Feature dimension of the vision model (e.g., gemma: 1152, openclip: 1024, qwen2.5: 1280)."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Input image resolution."
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=12,
        help="Number of transformer layers used in the aligner."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number for the web server."
    )

    return parser.parse_args()


# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    args = parse_args()

    print(f"[INFO] Loading models with:")
    print(f"  Vision model: {args.vision_model}")
    print(f"  Checkpoint:   {args.checkpoint_path}")
    print(f"  Feature dim:  {args.feature_dim}")
    print(f"  Image size:   {args.image_size}")
    print(f"  Layers:       {args.layers}")

    vision_encoder, aligner_net, vae_ref = load_models(
        args.vision_model,
        args.checkpoint_path,
        args.image_size,
        args.feature_dim,
        args.layers
    )

    app.state.vision_encoder = vision_encoder
    app.state.aligner_net = aligner_net
    app.state.vae_ref = vae_ref

    demo = build_gradio_ui(vision_encoder, aligner_net, vae_ref)
    app = gr.mount_gradio_app(app, demo, path="/")

    server_name = "127.0.0.1"
    share_token = secrets.token_urlsafe(32)
    share_url = networking.setup_tunnel(
        local_host=server_name,
        local_port=args.port,
        share_token=share_token,
        share_server_address=None,
        share_server_tls_certificate=None,
    )
    print(f"Share URL: {share_url}")
    uvicorn.run(app, host=server_name, port=args.port)