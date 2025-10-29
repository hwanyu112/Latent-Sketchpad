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
from data.dataset import MultimodalEvalDataset, load_models, decode_img, MultimodalDataset, load_image
from model.uni_qwen import UniQwenForConditionalGeneration
from model.uni_gemma import GemmaGenForConditionalGeneration
from gen_utils import untie_embeddings

from azure.identity import AzureCliCredential, get_bearer_token_provider
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider

from openai import AzureOpenAI
import openai
import time
import base64
import threading

def encode_image(image_path, image_format="png", target_size=(768, 768)):
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path.convert("RGB")

    image = image.resize(target_size, Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    image.save(buffered, format=image_format.upper())
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    return f"data:image/{image_format.lower()};base64,{base64_image}"

def llm_sample(messages, client, model, stop=None, *arg):
    success = False
    while not success:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                stop=stop,
                temperature=0.0,
            )
            response = completion.choices[0].message.content
            success = True
        except openai.AuthenticationError as e:
            print(f"AuthenticationError occurred: {e}")
            if hasattr(e, "status_code") and e.status_code == 401:
                time.sleep(5)
        except openai.RateLimitError as e:
            print(f"RateLimitError occurred: {e}")
            if hasattr(e, "status_code") and e.status_code == 429:
                time.sleep(5)
        except openai.BadRequestError as e:
            print(f"BadRequestError occurred: {e}")
            if hasattr(e, "status_code") and e.status_code == 400:
                time.sleep(5)
        except openai.InternalServerError as e:
            print(f"InternalServerError occurred: {e}")
            time.sleep(10)
    return response

def launch_ls(model, model_name, input_ids, attention_mask, pixel_values, token_type_ids,
              image_grid_thw, max_new_tokens, device="cuda"):
    with torch.inference_mode():
        if model_name.lower() == "gemma":
            generated_ids, image_embeds, image_vit_feats = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                pixel_values=pixel_values.to(device),
                token_type_ids=token_type_ids.to(device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        elif model_name.lower() == "qwen":
            generated_ids, image_embeds, image_vit_feats = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                pixel_values=pixel_values.to(device),
                image_grid_thw=image_grid_thw.to(device),
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )


model_name = "MODEL_NAME" #"gpt-4o_2024-11-20"

model_config = {
    "gpt-4o": {
        "config_list": [{
            "model": "gpt-4o",  # o3
            "tags": ["gpt-4o"],
            "api_type": "azure",
            "base_url": "BASE_URL",
            "azure_ad_token_provider": get_bearer_token_provider(AzureCliCredential(), "URL"),
            "api_version": "2024-12-01-preview",
            "temperature": 0,
            "top_p": 1,
            "stop": None,
        }]
    }
}

config = model_config[model_name]

reasoning_client = AzureOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
summary_client = AzureOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
prompt = '''
### 🧩 **Task Instruction: Maze Reasoning and Action Planning**

**Task Objective:**
From the maze in the input image `<image>`, output a valid action sequence that moves from the **start (green arrow)** to the **end (red circle)**.
Black lines = **walls** (impassable).
White areas = **open path** (traversable).
The agent must stay entirely inside the white path and must never cross black walls.

---

### 🧭 **Action Vocabulary**

Every step in the solution must be expressed using **only** these actions:

* **"go forward"** — Move straight in the current facing direction **along a continuous white corridor** until you reach the next corner, turn, T-junction, or intersection.

  * You must merge any uninterrupted straight stretch of path into a single `"go forward"` action.
  * You cannot partially stop in the middle of a straight hallway unless it meets another path.

* **"turn left"** — Rotate 90° left **in place**. This changes only orientation, not location.

* **"turn right"** — Rotate 90° right **in place**. This changes only orientation, not location.

No other actions are allowed.

---

### 🧠 **Reasoning and Output Requirements**

1. The model must reason through the maze **in segments**, where each segment covers **multiple consecutive actions** (typically 3–5 actions).

   * End the segment with an action summary in exactly this format (on its own line):

    The actions of this part are <actions>action_1, action_2, ..., action_n</actions>
    
2. Ensure each segment includes more than 3 actions before summarizing. 
'''

rewrite_prompt = '''
Please rewrite the following maze reasoning segments into fluent and natural language, following the style below. Keep the logical flow and end with the action list using the format `<actions>...</actions>`.

### Example 1 (Start):
Now, let's reason through the next 3 steps.

At the maze's starting point, a straight path extends ahead, leading into the maze. Continuing along this route, a left turn appears, directing movement deeper into the maze. Based on the visible layout, the next actions should be to move forward, then turn left, and continue forward.

The actions of this part are <actions>go forward, turn left, go forward</actions>

### Example 2 (Middle Segment):
Now, let's reason through the next 4 steps.

The path begins with a right turn, followed by a forward passage leading to another left corner. This section suggests turning right, proceeding forward, then turning left to continue further inside.

The actions of this part are <actions>turn right, go forward, turn left, go forward</actions>

---

Output the rewritten segments directly without any additional commentary or meta-text.
Now rewrite the following: 

'''

system_prompt = '''
You are a helpful assistant that rewrites maze navigation reasoning into clear, fluent English. 
For each segment, output a concise paragraph that describes the navigation path step by step 
using natural and consistent sentence patterns.

Follow these specific instructions:

1. **Start segments** (i.e., beginning from the green arrow):
   - Begin with: `Now, let's reason through the next N steps.`
   - Then describe the path using this pattern:
     "At the maze's starting point, a straight path extends ahead..."
     Follow with natural descriptions of the path's direction and transitions.
   - End with the action summary in this format:
     `<actions>go forward, turn left, go forward</actions>`

2. **Middle segments** (i.e., all other cases):
   - Begin with: `Now, let's reason through the next N steps.`
   - Then **always begin the description with**:
     "The path begins with..."
     Use smooth connectors like “followed by”, “which leads to”, “ending with”, etc.
   - End with the action summary in this format:
     `<actions>turn right, go forward, turn left, go forward</actions>`

General guidelines:
- Describe the visual navigation clearly and smoothly.
- Use varied but natural sentence flow, similar to the examples below.
- Do not include commentary, meta-text, or anything outside the required structure.
'''

def match_actions_prefix(actions, label):
    if len(actions) > len(label):
        return False

    for i in range(len(actions)):
        if actions[i] != label[i]:
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Multimodal Model Evaluation using Trainer")
    parser.add_argument("--model_path", type=str, default="qwen", help="Path to the model checkpoint")
    parser.add_argument(
        "--decoder_path",
        type=str,
        required=True,
        help="Path of the pretrained Sketch Decoder.",
    )
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
    json_output_path = os.path.join(args.output_dir, args.json_output_file)
    images_dir = os.path.join(args.output_dir, args.image_folder)
    vit_feature_dir = os.path.join(args.output_dir, "vit_features")
    if not os.path.exists(vit_feature_dir):
        os.makedirs(vit_feature_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    if "gemma" in args.model_path.lower(): 
        model_name = "gemma"
        model = GemmaGenForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,  # or use fp16 if bf16 is unsupported
            attn_implementation="flash_attention_2",
            #attn_implementation="eager",
        ).to('cuda')
        boi_token = "<start_of_image>"
        untie_embeddings(model)
        image_size = 896
    elif "qwen" in args.model_path.lower():
        model_name = "qwen"
        model = UniQwenForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,  # or use fp16 if bf16 is unsupported
            # attn_implementation="eager",
            attn_implementation="flash_attention_2",
        ).to('cuda')
        boi_token = "<|vision_start|><|image_pad|><|vision_end|>"
        image_size = 448
    aligner_net, vae_ref = load_models(model.device, args.decoder_path, feature_dim=model.config.vision_config.hidden_size)
    model.eval()
    print(f"Model loaded from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/gemma-3-12b-it" if model_name == "gemma" else "/path/to/Qwen2.5-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained("/path/to/gemma-3-12b-it" if model_name == "gemma" else "/path/to/Qwen2.5-VL-7B-Instruct")
   
    # print(f"Loaded {len(test_dataset)} samples from {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_file_name = f"{model_name}-ls.json"
    output_file_path = os.path.join(args.output_dir, output_file_name)
    if os.path.exists(output_file_path):
        with open(output_file_path, "r", encoding="utf-8") as f:
            new_data = json.load(f)
    else:
        new_data = []
    img_emb_path = os.path.join(args.output_dir, "img_embeds")
    text_emb_path = os.path.join(args.output_dir, "text_embeds")
    if not os.path.exists(img_emb_path):
        os.makedirs(img_emb_path)
    if not os.path.exists(text_emb_path):
        os.makedirs(text_emb_path)
    os.environ["IMG_EMB_PATH"] = img_emb_path

    for idx in tqdm(range(len(data))):
        # batch = test_dataset[idx]
        if idx < len(new_data):
            continue
        max_new_tokens = 2000
        new_sample = deepcopy(data[idx])
        os.environ["SAMPLE_IDX"] = str(idx)
        os.environ["FILLING_PATH"] = ""
        
        img_path1 = os.path.join("/path/to/image/", data[idx]["input_img"][0])
        input_images1 = encode_image(img_path1, image_format="png")
        
        messages = [
            {"role": "system", "content": "You are an AI assistant that must strictly follow the user's instructions exactly as given."},
            {"role": "user", 
            "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": input_images1},
                            "detail": "high"
                        }
                    ]
                    }
            
        ]
        resp = llm_sample(messages, reasoning_client, model_name, stop=["</actions>"])
        actions_part = resp[resp.find("<actions>") + len("<actions>") : ]
        full_actions = [action.strip() for action in actions_part.split(',')]
        correct = match_actions_prefix(full_actions, new_sample["label_actions"]) and (len(full_actions) >= 3)
        API_round = 0
        full_resp = resp + "</actions>"
        actions = full_actions.copy()
        while correct:
            if actions[-1] != "go forward":
                resp = resp + ", go forward</actions>" 
                actions.append("go forward")
            else:
                resp = resp + "</actions>"
            
            # Rewrite the reasoning process
            rewrite_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", 
                "content": [
                            {"type": "text", "text": rewrite_prompt + resp}
                        ]
                        }
                
            ]

            rewrite_resp = llm_sample(rewrite_messages, summary_client, "gpt-4o")

            match = re.search(r"<actions>(.*?)</actions>", rewrite_resp)
            if match:
                actions_str = match.group(1).strip()
                rewrite_actions = [a.strip() for a in actions_str.split(",")]
            else:
                print("No actions found in the rewritten response.")
                break
            if rewrite_actions != actions:
                print("****************************")
                print("Actions Changed after Rewriting!")
                print("Original Actions:", actions_part)
                print("Rewritten Actions:", actions_str)
                break
            
            if API_round == 0:
                full_resp = resp
                raw_text = new_sample["input_text"].replace("<image>", boi_token) + rewrite_resp
                model_input_images = [load_image(os.path.join("/path/to/image", p), image_size=image_size) for p in new_sample["input_img"]]

                orig = processor(text=raw_text,
                                    images=model_input_images,
                                    return_tensors="pt").to(dtype=torch.bfloat16)
                input_ids = orig["input_ids"]          
                attention_mask = orig["attention_mask"]
                pixel_values = orig["pixel_values"]             
                token_type_ids = orig.get("token_type_ids", None)
                image_grid_thw = orig.get("image_grid_thw", None)
                worker = threading.Thread(
                    target=launch_ls,
                    args=(
                        model,
                        model_name,
                        input_ids,
                        attention_mask,
                        pixel_values,
                        token_type_ids,
                        image_grid_thw,
                        max_new_tokens,
                        "cuda",
                    ),
                    daemon=True,  # daemon表示主进程结束时，这个线程会被杀掉
                )
                worker.start()
            else:
                orig = processor(text=rewrite_resp,
                                    images=model_input_images,
                                    return_tensors="pt").to(dtype=torch.bfloat16)
                input_ids = orig["input_ids"]
                input_ids_path = os.path.join(text_emb_path, f"text_input_ids_{idx}_{API_round}.pt")
                torch.save(input_ids, input_ids_path)
                os.environ["FILLING_PATH"] = input_ids_path
            
            save_path = os.path.join(img_emb_path, f"img_embeds_{idx}_{API_round}.pt")
            start_time = time.time()
            while True:
                if os.path.exists(save_path):
                    print("File exist:", save_path)
                    time.sleep(3) 
                    img_embeds = torch.load(save_path)
                    break

                elapsed = time.time() - start_time
                if elapsed > 600:
                    print("Timeout waiting for image embeddings.")
                    img_embeds = None
                    break

                time.sleep(2)  
            if img_embeds is not None:
                decoded = decode_img(img_embeds.to('cuda'), aligner_net, vae_ref, 'cuda')  
                pil_img = transforms.ToPILImage()(decoded)
                img_name = f"output_img-{idx}_{API_round}.png"
                pil_img.save(os.path.join(images_dir, img_name))
                next_input_images = encode_image(pil_img, image_format="png")
                messages.extend([
                    {"role": "assistant", "content": resp},
                    {
                        "role": "user",
                        "content": [
                            # {"type": "text", "text": "Continue the reasoning based on the latest image and follow the initial instructions. The path ahead is blocked by black walls, so you must determine the next turning direction (left or right) based on the wall positions and the open white corridors."},
                            {
                                "type": "image_url",
                                "image_url": {"url": next_input_images},
                                "detail": "high"
                            }
                        ]
                    }
                ])
                resp = llm_sample(messages, reasoning_client, model_name, stop=["</actions>"])
                actions_part = resp[resp.find("<actions>") + len("<actions>") : ]
                actions = [action.strip() for action in actions_part.split(',')]
                full_actions.extend(actions)
                correct = match_actions_prefix(full_actions, new_sample["label_actions"]) and (len(actions) >= 3)
                full_resp = full_resp + resp + "</actions>"
                API_round += 1
            else:
                break
        new_sample["generated_actions"] = full_actions
        new_sample["model_output"] = full_resp
        new_sample["idx"] = idx
        os.environ["FILLING_PATH"] = "END"
        new_data.append(new_sample)

 
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    main()