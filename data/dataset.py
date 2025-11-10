import json
import os
import io
import base64
import re
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from diffusers.models import AutoencoderKL
from decoder.aligner.dense_aligner import ClipToLatentAligner
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
# from qwen_vl_utils import process_vision_info
from typing import Dict, List, Optional, Union, Any
import requests
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gen_utils import left_padding

sep_token = "<unused6>"
IGNORE_INDEX = -100
BOI_TOKEN_Gemma3 = "<start_of_image>"
BOI_TOKEN_Qwen = "<|vision_start|><|image_pad|><|vision_end|>"
IMAGE_SIZE = 224
GRID_SIZE = IMAGE_SIZE // 8
LAYERS = 12

def find_subseq(haystack: torch.Tensor, needle: list[int]) -> int:
    """
    在 1-D tensor haystack 中查找 needle（Python list）第一次出现的位置。
    若找不到返回 -1。
    """
    n = len(needle)
    for i in range(haystack.size(0) - n + 1):
        if torch.equal(haystack[i:i+n], torch.tensor(needle, device=haystack.device)):
            return i
    return -1

def _keep_only_first_image_token(text: str) -> str:
    """
    在文本中保留第一个 <image>，删除其余所有 <image>。
    若文本中没有 <image>，原样返回。
    """
    cnt = 0
    def _repl(m):
        nonlocal cnt
        cnt += 1
        return "<image>" if cnt == 1 else ""   # 第一个保留，其余删掉
    return re.sub(r"<image>", _repl, text)

def _remove_all_image_tokens(text: str) -> str:
    """删除文本中所有 <image>。"""
    return re.sub(r"<image>", "", text)

def load_image(image_path: str, image_size: int = 896):
    try:
        if image_path.startswith('http'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path).resize((image_size, image_size))
        return image
    except Exception:
        print(f'Error occurred when dealing with {image_path}')
        raise Exception

def decode_img(image_features, aligner_net, vae_ref, device):
    inp = image_features.to(torch.float32)
    # ensure batch dim
    if inp.ndim < 3:
        inp = inp.unsqueeze(0)
    with torch.no_grad():
        mask = torch.zeros(inp.shape[:2], dtype=torch.bool).to(device)
        _, latent_data = aligner_net.encode(inp, mask)
        latent_tensor = latent_data.latent_dist.mode()
        decoded = vae_ref.decode(latent_tensor).sample
        tensor = (decoded.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
        
    return tensor.cpu()

def load_models(device, checkpoint_path, feature_dim):
    vae_ref = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae_ref.eval()

    aligner_net = ClipToLatentAligner(None, feature_dim, 512, GRID_SIZE, LAYERS).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace('aligner_net.', ''): v for k, v in checkpoint['state_dict'].items()}
    aligner_net.load_state_dict(state_dict)
    aligner_net.eval()

    return aligner_net, vae_ref

class MultimodalDataset(Dataset):
    """
    Dataset class for multimodal training with Gemma 3.
    Handles JSON data with text and image inputs.
    """
    
    def __init__(
        self,
        tokenizer,
        processor,
        model,
        checkpoint_path, 
        feature_dim,
        json_file: str,
        image_dir: str = "/mnt/maze/reasoning_maze/",
        image_size: int = 896,
        augment: bool = False,
        stage1: bool = False,
        model_name: str = "Gemma3",
        image_token_index=262144,
        boi_id=159999,
        eoi_id=160000,
        ignore_image: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            json_file: Path to the JSON file containing the data
            tokenizer_name_or_path: Name or path of the tokenizer
            processor_name_or_path: Name or path of the processor for image processing
            image_dir: Directory containing the images
            max_length: Maximum sequence length for tokenization
            image_size: Size to resize images to
        """
        self.json_file = json_file
        self.image_dir = image_dir
        self.image_size = image_size
        
        # Load the data
        self.data = []
        for file in json_file:
            with open(file, 'r', encoding='utf-8') as f:
                self.data.extend(json.load(f))
            
        # Initialize tokenizer and processor
        self.tokenizer = tokenizer
        self.processor = processor
        self.model = model
        self.augment = augment
        self.stage1 = stage1
        aligner_net, vae_ref = load_models(model.device, checkpoint_path, feature_dim)
        self.aligner_net = aligner_net
        self.vae_ref = vae_ref
        self.device = model.device
        self.model_name = model_name
        self.image_token_index = image_token_index.cpu()
        self.boi_id = boi_id
        self.eoi_id = eoi_id
        self.ignore_image = ignore_image

        if "gemma" in model_name.lower():
            self.boi_token = BOI_TOKEN_Gemma3
            self.image_size = 896
        elif "qwen" in model_name.lower():
            self.boi_token = BOI_TOKEN_Qwen
            self.image_size = 448

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing input_ids, attention_mask, pixel_values, and labels
        """
        item = self.data[idx]

        if random.random() < 0.5 and self.stage1:
            # text-only
            item["input_text"] = _keep_only_first_image_token(item["input_text"])
            item["label_text"] = _remove_all_image_tokens(item["label_text"])
            item["label_img"] = []
            item["input_img"] = item["input_img"][:1]

        # Process text input
        input_text = item["input_text"].replace("<image>", self.boi_token)
        label_text = item["label_text"].replace("<image>", self.boi_token)

        text = input_text + label_text
        prompt = input_text

        raw_text = ''
        if isinstance(text, list):
            raw_text = self.tokenizer.eos_token.join(text)
        elif isinstance(text, str):
            if text.endswith('The inference process is complete.'):
                raw_text = text + self.tokenizer.eos_token
            else:
                raw_text = text
        else:
            raise NotImplementedError

        # Process images
        if self.augment:
            input_images = []
            label_images = [load_image(os.path.join(self.image_dir, img_path), image_size=self.image_size) for img_path in item["label_img"]]
            i = 0
            for img_path in item["input_img"]:
                img = load_image(os.path.join(self.image_dir, img_path), image_size=self.image_size)
                if i != 0:
                    aug_n = random.randint(0, 2)
                    for _ in range(aug_n):
                        aug_input = self.processor(text=self.boi_token, images=img, return_tensors='pt').to(dtype=torch.bfloat16)
                        pixel_values = aug_input['pixel_values']
                        if self.model_name.lower() == 'gemma':
                            image_features = self.model.get_vit_features(pixel_values=pixel_values.to(self.device))
                        elif self.model_name.lower() == 'qwen':
                            image_grid_thw = aug_input['image_grid_thw']
                            image_features = self.model.get_vit_features(pixel_values=pixel_values.to(self.device), image_grid_thw=image_grid_thw.to(self.device))
                        decoded_images = decode_img(image_features, self.aligner_net, self.vae_ref, self.device)
                        img = transforms.ToPILImage()(decoded_images).resize((self.image_size, self.image_size))
                input_images.append(img)
                i += 1
        else:
            input_images = [load_image(os.path.join(self.image_dir, img_path), image_size=self.image_size) for img_path in item["input_img"]]
            label_images = [load_image(os.path.join(self.image_dir, img_path), image_size=self.image_size) for img_path in item["label_img"]]

        images_kwargs = {
            "do_resize": False,
            }
        # Process text and prompt
        text_dict = self.processor(text=raw_text, images=input_images+label_images, return_tensors='pt', images_kwargs=images_kwargs).to(
            dtype=torch.bfloat16
        )
        prompt_dict = self.processor(text=prompt, images=input_images, return_tensors='pt', images_kwargs=images_kwargs).to(
            dtype=torch.bfloat16
        )

        labels = text_dict['input_ids'].clone().squeeze()
        # mask non-assistant input
        input_len = len(prompt_dict['input_ids'].squeeze()) 
        labels[: input_len] = IGNORE_INDEX
        labels[labels==self.boi_id] = IGNORE_INDEX
        labels[labels==self.eoi_id] = IGNORE_INDEX
        if self.ignore_image:
            labels[labels==self.image_token_index] = IGNORE_INDEX

        if text_dict['pixel_values'].shape[0] == 0:
            raise ValueError(f"Empty image tensor for index {idx}, {len(input_images)} images found.")
        
        
        # Prepare the output
        if self.model_name.lower() == 'gemma':
            return {
                "input_ids": text_dict['input_ids'].squeeze(),
                "attention_mask": text_dict["attention_mask"].squeeze(0),
                "labels": labels,
                "token_type_ids": text_dict["token_type_ids"].squeeze(0),
                "pixel_values": text_dict["pixel_values"],
            }
        elif self.model_name.lower() == 'qwen':
            return {
                "input_ids": text_dict['input_ids'].squeeze(),
                "attention_mask": text_dict["attention_mask"].squeeze(0),
                "labels": labels,
                "pixel_values": text_dict["pixel_values"],
                "image_grid_thw": text_dict["image_grid_thw"],
            }


class MultimodalEvalDataset(Dataset):
    """
    Dataset class for multimodal training with Gemma 3.
    Handles JSON data with text and image inputs.
    """
    
    def __init__(
        self,
        tokenizer,
        processor,
        json_file: str,
        image_dir: str = "/path/to/reasoning_maze/",
        image_size: int = 896,
        text_only: bool = False,
        is_original_model: bool = False,
        model_name: str = "Gemma3"
    ):
        """
        Initialize the dataset.
        
        Args:
            json_file: Path to the JSON file containing the data
            tokenizer_name_or_path: Name or path of the tokenizer
            processor_name_or_path: Name or path of the processor for image processing
            image_dir: Directory containing the images
            max_length: Maximum sequence length for tokenization
            image_size: Size to resize images to
        """
        self.json_file = json_file
        self.image_dir = image_dir
        self.image_size = image_size
        
        # Load the data
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        # Initialize tokenizer and processor
        self.tokenizer = tokenizer
        self.processor = processor
        self.text_only = text_only
        self.is_original_model = is_original_model
        self.model_name = model_name

        if "gemma" in model_name.lower():
            self.boi_token = BOI_TOKEN_Gemma3
            self.image_size = 896
            self.image_token_id = 262144
        elif "qwen" in model_name.lower():
            self.boi_token = BOI_TOKEN_Qwen
            self.image_size = 448
            self.image_token_id = 151655
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing input_ids, attention_mask, pixel_values, and labels
        """
        item = self.data[idx]

        raw_text = item["input_text"].replace("<image>", self.boi_token)
        # raw_text = raw_text + item["label_text"]
        input_images = [load_image(os.path.join(self.image_dir, p), image_size=self.image_size) for p in item["input_img"]]

        orig = self.processor(text=raw_text,
                            images=input_images,
                            return_tensors="pt").to(dtype=torch.bfloat16)

        # Single batch: remove batch dim for easier handling later
        input_ids = orig["input_ids"].squeeze(0)          # shape [L]
        attention_mask = orig["attention_mask"].squeeze(0)
        pixel_values = orig["pixel_values"].squeeze(0)               
        token_type_ids = orig.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(0)
        image_grid_thw = orig.get("image_grid_thw", None)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.squeeze(0)

        # Prepare the output
        if self.model_name.lower() == 'gemma':
            return {
                "input_ids": input_ids.squeeze(),
                "attention_mask": attention_mask.squeeze(0),
                "token_type_ids": token_type_ids,
                "pixel_values": pixel_values.squeeze(0)
            }
        elif self.model_name.lower() == 'qwen':
            return {
                "input_ids": input_ids.squeeze(),
                "attention_mask": attention_mask.squeeze(0),
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

class DataCollatorForInputs:
    def __init__(self, pad_token_id=IGNORE_INDEX, model_name="gemma"):
        self.pad_token_id = pad_token_id  
        self.model_name = model_name

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [torch.tensor(f["labels"]) for f in features if 'labels' in f]
        
        # get the last image label
        pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)
        """  # Pad to max length in batch
        max_len = max(e.shape[0] for e in inputs_embeds)
        embed_dim = inputs_embeds[0].shape[1]

        # Pad inputs_embeds manually to shape [B, max_len, D]
        padded_inputs_embeds = torch.stack([
            torch.cat([e, e.new_zeros(max_len - e.size(0), embed_dim)]) for e in inputs_embeds
        ])

        # Pad attention masks to [B, max_len]
        attention_masks = torch.stack([
            torch.cat([torch.ones(e.shape[0]), torch.zeros(max_len - e.shape[0])]) for e in inputs_embeds
        ]).long() """

        # Pad labels to [B, max_len]
        padded_input_ids = left_padding(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_labels = left_padding(labels, batch_first=True, padding_value=self.pad_token_id)
        attention_masks = padded_input_ids.ne(self.pad_token_id)
        if self.model_name.lower() == 'gemma':
            token_type_ids = [torch.tensor(f["token_type_ids"]) for f in features]
            padded_token_type_ids = left_padding(token_type_ids, batch_first=True, padding_value=self.pad_token_id)
        
            return {
                "input_ids": padded_input_ids,
                "attention_mask": attention_masks,
                "labels": padded_labels if len(padded_labels) > 0 else None,
                "token_type_ids": padded_token_type_ids,
                "pixel_values": pixel_values,
            }
        elif self.model_name.lower() == 'qwen':
            image_grid_thw = [f["image_grid_thw"] for f in features]
            image_grid_thw = torch.cat(image_grid_thw, dim=0)
            return {
                    "input_ids": padded_input_ids,
                    "attention_mask": attention_masks,
                    "labels": padded_labels if len(padded_labels) > 0 else None,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                }