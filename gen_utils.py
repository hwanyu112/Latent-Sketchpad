import os
import sys
import uuid
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, TypedDict, Union
import yaml
import shutil
import re

import torch
from torch.nn.utils.rnn import pad_sequence
from typing_extensions import NotRequired
import io, base64
from PIL import Image
sys.path.append(os.path.dirname(__file__))

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

img_token_start_id = 256000

def left_padding(sequences, batch_first=True, padding_value=0):
    """
    Left pads a list of 1D tensors using flipping.
    
    Args:
        sequences (list of Tensors): List of 1D tensors to pad.
        batch_first (bool): Whether the returned tensor should have batch as the first dimension.
        padding_value (int or float): Value to use for padding.
    
    Returns:
        Tensor: Left-padded tensor.
    """
    # Step 1: Reverse each sequence along its only dimension (dim=0)
    reversed_seqs = [seq.flip(0) for seq in sequences]
    
    # Step 2: Right pad the reversed sequences (using pad_sequence)
    padded_reversed = pad_sequence(reversed_seqs, batch_first=batch_first, padding_value=padding_value)
    
    # Step 3: Flip the padded tensor back along the sequence dimension.
    # When batch_first=True, sequence dimension is 1; otherwise it's 0.
    seq_dim = 1 if batch_first else 0
    padded = padded_reversed.flip(seq_dim)
    
    return padded

def untie_embeddings(model):
    """
    Untie the embeddings of the model.
    """
    import copy
    if not model.config.text_config.tie_word_embeddings:
        print("Embeddings are already untied.")
        return

    # Assuming `model` is your pre-trained model
    # Create a new, independent lm_head layer.
    new_lm_head = torch.nn.Linear(model.config.text_config.hidden_size, model.config.text_config.vocab_size, bias=False)

    # Optionally, copy the existing weights from embed_tokens into the new lm_head.
    new_lm_head.weight.data.copy_(copy.deepcopy(model.language_model.model.embed_tokens.weight.data))

    # Replace the tied lm_head with the new independent one.
    model.language_model.lm_head = new_lm_head.to(torch.bfloat16).to(model.device)
    model.config.text_config.tie_word_embeddings = False
    model.language_model._tied_weights_keys = None


def pil_to_base64(image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

_re_checkpoint = re.compile(r"checkpoint-(\d+)$")
KEY_EXTS = {".pt", ".pth", ".safetensors"}

def _list_checkpoints(folder):
    """List all checkpoint directories matching the naming rule, sorted by step in ascending order"""
    if not os.path.isdir(folder):
        return []
    ckpts = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isdir(path) and _re_checkpoint.match(name):
            ckpts.append(path)
    ckpts.sort(key=lambda p: int(_re_checkpoint.search(os.path.basename(p)).group(1)))
    return ckpts

def _list_key_files(ckpt_dir):
    files = set()
    for dp, _, fns in os.walk(ckpt_dir):
        for fn in fns:
            _, ext = os.path.splitext(fn)
            if ext.lower() in KEY_EXTS:
                files.add(fn)  # 只取文件名
    return files

def _safe_remove(path):
    try:
        shutil.rmtree(path)
    except Exception:
        try:
            os.rename(path, path + ".to_delete")
            shutil.rmtree(path + ".to_delete", ignore_errors=True)
        except Exception:
            pass

def _is_complete(curr_ckpt, prev_ckpt):
    """Check whether curr_ckpt and prev_ckpt have the same set of key files"""
    return _list_key_files(curr_ckpt) == _list_key_files(prev_ckpt)

def get_last_checkpoint(folder):
    """
    Return the last available checkpoint:
    - If the latest checkpoint and the previous one have the same set of key files → return the latest
    - If not → delete the latest and continue rolling back
    - If there is only one checkpoint → skip the integrity check and return it directly (may require retraining from scratch)
    """
    ckpts = _list_checkpoints(folder)
    if not ckpts:
        return None

    while ckpts:
        if len(ckpts) == 1:
            return ckpts[-1]  # 只有一个就直接返回

        last = ckpts[-1]
        prev = ckpts[-2]

        if _is_complete(last, prev):
            return last
        else:
            _safe_remove(last)
            ckpts.pop()

    return None

