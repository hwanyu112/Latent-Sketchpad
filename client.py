#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client.py – call /reconstruct_image on the Align+VAE FastAPI app
Usage:
    python client.py --input path/to/input.jpg \
                     --output path/to/output.png \
                     --host 127.0.0.1 \
                     --port 7860 \
                     --mount /        # change to "/gradio" if you kept the Gradio mount
"""
import argparse
import os
import sys
import requests

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call /reconstruct_image endpoint and save result.")
    parser.add_argument("--input", "-i", default="/path/to/input.png", help="Path to the source image.") 
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860).")
    parser.add_argument(
        "--mount",
        default="/",
        help='Path prefix if the API is mounted under a sub-path (e.g. "/gradio");'
             ' keep "/" if you run uvicorn directly.',
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Basic sanity checks
    if not os.path.exists(args.input):
        sys.exit(f"[Error] Input image {args.input} does not exist.")
    if not args.output.lower().endswith(".png"):
        print("[Info] Forcing PNG extension on output file.")
        args.output += ".png"

    endpoint = f"http://{args.host}:{args.port}{args.mount.rstrip('/')}/reconstruct_image"
    print(f"[Info] Sending request to {endpoint}")

    try:
        with open(args.input, "rb") as f:
            response = requests.post(
                endpoint,
                files={"file": (os.path.basename(args.input), f, "image/*")},
                stream=True,
                timeout=60,
            )
    except requests.exceptions.RequestException as e:
        sys.exit(f"[Error] Failed to connect to server: {e}")

    if response.status_code != 200:
        sys.exit(f"[Error] Server returned {response.status_code}: {response.text}")

    # Stream response content to disk
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as out_f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                out_f.write(chunk)

    print(f"[Success] Reconstructed image saved to {args.output}")

if __name__ == "__main__":
    main()