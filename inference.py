import os
import json
import io
import re
from collections import defaultdict
from typing import Dict, Any, List, Optional
from copy import deepcopy
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import argparse
import shutil
from transformers import AutoTokenizer, AutoProcessor
from transformers import set_seed
from data.dataset import MultimodalEvalDataset, load_models, decode_img, MultimodalDataset
from model.uni_qwen import UniQwenForConditionalGeneration
from model.uni_gemma import GemmaGenForConditionalGeneration
from gen_utils import untie_embeddings

def main():
    parser = argparse.ArgumentParser(description="Multimodal Model Evaluation using Trainer")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--decoder_path", type=str, required=True, help="Path of the pretrained Sketch Decoder.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for evaluation results")
    parser.add_argument("--image_folder", type=str, default="generated_images", help="Folder name under output_dir to save generated images")
    parser.add_argument("--json_output_file", type=str, default="generated_outputs.json", help="JSON file (in output_dir) to store generated outputs")
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation ('cuda' or 'cpu')")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    set_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    images_dir = os.path.join(args.output_dir, args.image_folder)
    vit_feature_dir = os.path.join(args.output_dir, "vit_features")
    if not os.path.exists(vit_feature_dir):
        os.makedirs(vit_feature_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    if "gemma" in args.model_path.lower(): 
        model_name = "gemma"
        if os.path.exists(args.model_path):
            shutil.copy('/path/to/gemma-3-12b-it/preprocessor_config.json', args.model_path)

        model = GemmaGenForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,  # or use fp16 if bf16 is unsupported
            attn_implementation="flash_attention_2"
        ).to('cuda')
        untie_embeddings(model)
    elif "qwen" in args.model_path.lower():
        model_name = "qwen"
        if os.path.exists(args.model_path):
            shutil.copy('/path/to/Qwen2.5-VL-7B-Instruct/preprocessor_config.json', args.model_path)
    
        model = UniQwenForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,  # or use fp16 if bf16 is unsupported
            # attn_implementation="eager",
            attn_implementation="flash_attention_2",
        ).to('cuda')
    aligner_net, vae_ref = load_models(model.device, args.decoder_path, feature_dim=model.config.vision_config.hidden_size)
    model.eval()
    print(f"Model loaded from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/gemma-3-12b-it" if model_name == "gemma" else "/path/to/Qwen2.5-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained("/path/to/gemma-3-12b-it" if model_name == "gemma" else "/path/to/Qwen2.5-VL-7B-Instruct")
    test_dataset = MultimodalEvalDataset(tokenizer, processor, args.data_path, image_dir="/path/to/reasoning_maze/", is_original_model=False, model_name=model_name)   
    print(f"Loaded {len(test_dataset)} samples from {args.data_path}")

    batch = test_dataset[1]
    max_new_tokens = 2500

    with torch.inference_mode():
        if model_name.lower() == "gemma":
            generated_ids, image_embeds, image_vit_feats = model.generate(
                            input_ids=batch["input_ids"].unsqueeze(0).to('cuda'),
                            attention_mask=batch["attention_mask"].unsqueeze(0).to('cuda'),
                            pixel_values=batch["pixel_values"].unsqueeze(0).to('cuda'),
                            token_type_ids=batch["token_type_ids"].unsqueeze(0).to('cuda'),
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                        )
            
        elif model_name.lower() == "qwen":
            generated_ids, image_embeds, image_vit_feats = model.generate(
                                input_ids=batch["input_ids"].unsqueeze(0).to('cuda'),
                                attention_mask=batch["attention_mask"].unsqueeze(0).to('cuda'),
                                pixel_values=batch["pixel_values"].unsqueeze(0).to('cuda'),
                                image_grid_thw=batch["image_grid_thw"].unsqueeze(0).to('cuda'),
                                do_sample=False,
                                max_new_tokens=max_new_tokens,
                            )
               
    output_token_ids = (
            generated_ids.to(dtype=batch["input_ids"].dtype)
            .detach()
            .cpu()
            .numpy()
        )
    input_length = batch["input_ids"].shape[0]
    sequence = output_token_ids[0]
    input_token_ids = sequence[:input_length]
    response_token_ids = sequence[input_length:]
    response = tokenizer.decode(response_token_ids.tolist(), skip_special_tokens=True)
    decoded_input = tokenizer.decode(input_token_ids.tolist(), skip_special_tokens=True)
    
    print(f"Input: {decoded_input}")
    print(f"Response: {response}")

    if image_embeds is not None:
         for j in range(image_vit_feats.size(0)):
            feat = image_vit_feats[j].to('cuda')          
            decoded = decode_img(feat, aligner_net, vae_ref, 'cuda')  
            pil_img = transforms.ToPILImage()(decoded)

            img_name = f"output_img-{j}.png"
            pil_img.save(os.path.join(images_dir, img_name))

if __name__ == "__main__":
    main()