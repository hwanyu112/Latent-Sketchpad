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
from gen_utils import untie_embeddings, left_padding
from transformers import set_seed
from data.dataset import MultimodalEvalDataset, load_models, decode_img
from model.uni_gemma import GemmaGenForConditionalGeneration, Gemma3ForConditionalGeneration
from model.uni_qwen import UniQwenForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from evaluator import text_evaluator, image_evaluator, evaluate_layout
import requests

############################################
# Evaluation Functions: Text and Image Eval
############################################
SIZE_RE = re.compile(r"size_(\d+)_(\d+)")

def text_evaluation(
    json_file,
    text_evaluator,
    output_dir,
    label_file,
):
    """
    Evaluate generated texts, save per-sample augmented JSON, and
    report metrics both globally and grouped by maze size (e.g. 5×4).
    """

    # ---------- Read Data ----------
    with open(json_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    with open(label_file, 'r', encoding='utf8') as f:
        labels = json.load(f)

    # ---------- Global Accumulator ----------
    global_sum = defaultdict(float)
    global_cnt = 0
    global_entirely_correct = 0
    global_success = 0

    # ---------- Accumulator for Each Size ----------
    # dict: size_key -> dict(metric_name -> sum/计数)
    size_stats = defaultdict(lambda: defaultdict(float))
    size_cnt   = defaultdict(int)
    size_entirely_correct = defaultdict(int)
    size_success = defaultdict(int)

    new_data = []

    for idx, sample in tqdm(enumerate(data), total=len(data), desc="Evaluating texts"):

        output_text   = sample["output_text"]
        label_actions = sample["label_actions"]

        # === Maze Size Key, in the Form "5x4" ===
        maze_path = labels[idx]["input_img"][0]
        m = SIZE_RE.search(maze_path)
        if not m:
            raise ValueError(f"Unrecognized maze path: {maze_path}")
        size_key = f"{m.group(1)}x{m.group(2)}"

        # Use answer_label for Samples with "Total Steps = 0"
        if labels[idx]["total_step"] == 0:
            answer_label = label_actions

        # ---------- Evaluation ----------
        res = text_evaluator(output_text, label_actions)
        acc         = res["accuracy"]            # entire seq accuracy
        st_acc      = res["state_accuracy"]      # next-state level acc
        next_len    = res["next_state_len"]
        correct_len = res["correct"]
        answer_seq  = res["answer_sequence"]

        # ---------- Global Accumulation ----------
        global_sum["accuracy"]          += acc
        global_sum["state_accuracy"]    += st_acc
        global_sum["next_state_len"]    += next_len
        global_sum["correct_len"]       += correct_len
        global_cnt += 1
        if st_acc == 1.0:
            global_entirely_correct += 1
        if acc == 1.0:
            global_success += 1

        # ---------- Per-Size Accumulation ----------
        stats = size_stats[size_key]
        stats["accuracy"]          += acc
        stats["state_accuracy"]    += st_acc
        stats["next_state_len"]    += next_len
        stats["correct_len"]       += correct_len
        size_cnt[size_key]         += 1
        if st_acc == 1.0:
            size_entirely_correct[size_key] += 1
        if acc == 1.0:
            size_success[size_key] += 1

        # ---------- answer accuracy ----------
        ans_corr = 0
        if answer_seq is not None and len(answer_seq) <= len(answer_label):
            for pa, ta in zip(answer_seq, answer_label):
                if pa == ta:
                    ans_corr += 1
                else:
                    break
        ans_acc = ans_corr / len(answer_label)

        global_sum["answer_accuracy"] += ans_acc
        stats["answer_accuracy"]      += ans_acc

        aug = deepcopy(sample)
        aug["action_sequences"] = res["entire_action_sequence"]
        aug["answer"] = [] if answer_seq is None else answer_seq
        new_data.append(aug)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    def _avg(d_sum, cnt):
        return {k: v / cnt if cnt else 0.0 for k, v in d_sum.items()}

    global_avg = _avg(global_sum, global_cnt)
    global_avg["entire_next_state_accuracy"] = global_entirely_correct / global_cnt if global_cnt else 0.0
    global_avg["success_rate"]               = global_success / global_cnt if global_cnt else 0.0

    size_avg = {}
    for s in size_stats.keys():
        avg = _avg(size_stats[s], size_cnt[s])
        avg["entire_next_state_accuracy"] = size_entirely_correct[s] / size_cnt[s]
        avg["success_rate"]               = size_success[s]         / size_cnt[s]
        size_avg[s] = avg

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("==== GLOBAL ====\n")
        f.write(f"Total samples: {global_cnt}\n")
        f.write(f"Average Progress Rate: {global_avg['accuracy']:.4f}\n")
        f.write(f"Average accuracy of next state: {global_avg['state_accuracy']:.4f}\n")
        f.write(f"Average predicted next state action length: {global_avg['next_state_len']:.2f}\n")
        f.write(f"Average correct predicted action length: {global_avg['correct_len']:.2f}\n")
        f.write(f"Average entire next state accuracy: {global_avg['entire_next_state_accuracy']:.2f}\n")
        f.write(f"Average Success Rate: {global_avg['success_rate']:.4f}\n\n")

        f.write("==== BY MAZE SIZE ====\n")
        for s, avg in sorted(size_avg.items()):
            f.write(f"\n-- Maze {s} ({size_cnt[s]} samples) --\n")
            f.write(f"  Progress Rate: {avg['accuracy']:.4f}\n")
            f.write(f"  Next-state accuracy: {avg['state_accuracy']:.4f}\n")
            f.write(f"  Pred next len: {avg['next_state_len']:.2f}\n")
            f.write(f"  Correct len: {avg['correct_len']:.2f}\n")
            f.write(f"  Entire next-state acc: {avg['entire_next_state_accuracy']:.2f}\n")
            f.write(f"  Success rate: {avg['success_rate']:.4f}\n")

    with open(os.path.join(output_dir, "results_by_size.json"), "w", encoding="utf-8") as f:
        json.dump(size_avg, f, ensure_ascii=False, indent=2)

def image_evaluation(json_file, image_evaluator, output_dir, label_file, images_dir, path_prefix="/path/to/reasoning_maze/"):
    """
    Evaluate generated images by reading their paths from a JSON file.
    
    Parameters:
        json_file (str): Path to JSON file containing outputs.
        image_evaluator (callable): Function to evaluate images.
        kwargs: Additional arguments for the evaluator.
    
    Returns:
        Evaluation results returned by image_evaluator.
    """
    with open(json_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    with open(label_file, 'r', encoding='utf8') as f:
        label = json.load(f)
    
    total_last_excess = 0
    total_last_reached = 0
    excess = []
    for idx, sample in tqdm(enumerate(data), total=len(data), desc="Evaluating images"):
        input_image_path = images_dir + '/' + sample["input_image_paths"][-1]
        if len(sample["output_image_paths"]) == 0:
            continue
        else:
            img_path_list = sample["output_image_paths"]
            # evaluate the last image
            label_image_path = path_prefix + label[idx]["rl_label_img"][-1]
            last_img = images_dir + '/' + img_path_list[-1]
            result = image_evaluator(input_path=input_image_path, output_path=last_img, label_path=label_image_path)
            if result["arrived"]:
                total_last_reached += 1
                if result["excess"]:
                    total_last_excess += 1
                    excess.append(idx)
            
    total_samples = len(data)
    avg_last_reached = (total_last_reached - total_last_excess) / total_samples if total_samples > 0 else 0.0
    results_path = os.path.join(output_dir, "image_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Average Visual Success Rate: {avg_last_reached:.4f}\n")
    

def append_jsonl(obj: dict, json_path: str):
    obj_str = json.dumps(obj, ensure_ascii=False, indent=4)

    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            f.write(obj_str)
            f.write("\n]")
        return

    with open(json_path, "r+", encoding="utf-8") as f:
        f.seek(0, io.SEEK_END)
        if f.tell() == 0:
            f.write("[\n" + obj_str + "\n]")
            return

        pos = f.tell() - 1
        while pos > 0:
            f.seek(pos)
            ch = f.read(1)
            if ch not in " \t\r\n":
                break
            pos -= 1

        if ch == "[":
            f.seek(pos + 1)                    
            f.write("\n" + obj_str + "\n]")
        elif ch == "]":
            f.seek(pos)                        
            f.truncate()                       
            f.write(",\n" + obj_str + "\n]")   
        else:
            raise RuntimeError(
                f"Unexpected file format near position {pos}: found '{ch}'."
            )

def load_done_indices(json_path: str, key: str = "sample_idx") -> set[int]:
    done = set()
    if not os.path.exists(json_path):
        return done

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)          
            for entry in data:
                if isinstance(entry, dict) and key in entry:
                    done.add(entry[key])
        except json.JSONDecodeError as e:
            f.seek(0)
            buf = ""
            for line in f:
                buf += line
                try:
                    item = json.loads(buf)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, list):
                    for entry in item:
                        if isinstance(entry, dict) and key in entry:
                            done.add(entry[key])
                    break
    return done


#############################################################
#                        Evaluation                         #
#############################################################
def run_evaluation(
    model,
    test_dataset,
    tokenizer,
    images_dir,
    jsonl_path,
    label_file,
    aligner_net,
    vae_ref,
    device,
    max_new_tokens=8192,
    batch_size=1,
    vit_output_dir=None,
    is_original_model=False,
    model_name="gemma" 
):

    done_set = load_done_indices(jsonl_path)
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    responses = []
    input_decoded_list = []
    idx = 0
    image_vit_features_paths = {}
    for batch in tqdm(dataloader, desc="Evaluating"):
        if all((idx + i) in done_set for i in range(len(batch["input_ids"]))):
            idx += len(batch["input_ids"])
            continue
        with torch.inference_mode():
            if model_name.lower() == "gemma":
                if is_original_model:
                    generated_ids = model.generate(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        max_new_tokens=2500,
                        do_sample=False,
                    )
                    image_embeds = image_vit_feats = None
                else:
                    generated_ids, image_embeds, image_vit_feats = model.generate(
                        input_ids=batch["input_ids"].to('cuda'),
                        attention_mask=batch["attention_mask"].to('cuda'),
                        pixel_values=batch["pixel_values"].to('cuda'),
                        token_type_ids=batch["token_type_ids"].to('cuda'),
                        # temperature = 0.75, top_p = 0.95, top_k = 64,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                    )
            elif model_name.lower() == "qwen":
                if is_original_model:
                    batch["input_ids"] = batch["input_ids"][0]
                    batch["attention_mask"] = batch["attention_mask"][0]
                    batch["pixel_values"] = batch["pixel_values"][0]
                    batch["image_grid_thw"] = batch["image_grid_thw"][0]
                    generated_ids = model.generate(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        pixel_values=batch["pixel_values"].to(device),
                        image_grid_thw=batch["image_grid_thw"].to(device),
                        max_new_tokens=1500,
                        do_sample=False,
                    )
                    image_embeds = image_vit_feats = None
                else:
                    generated_ids, image_embeds, image_vit_feats = model.generate(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                            pixel_values=batch["pixel_values"].to(device),
                            image_grid_thw=batch["image_grid_thw"].to(device),
                            do_sample=False,
                            max_new_tokens=max_new_tokens,
                        )

        output_token_ids_batch = (
            generated_ids.to(dtype=batch["input_ids"].dtype)
            .detach()
            .cpu()
            .numpy()
        )
        input_length = batch["input_ids"].shape[1]

        for sequence in output_token_ids_batch:
            input_token_ids = sequence[:input_length]
            response_token_ids = sequence[input_length:]

            response = tokenizer.decode(response_token_ids.tolist(), skip_special_tokens=True)
            decoded_input = tokenizer.decode(input_token_ids.tolist(), skip_special_tokens=True)

            responses.append(response)
            input_decoded_list.append(decoded_input)
            image_vit_features_paths[idx] = []
            if image_embeds is not None and image_embeds.shape[0] > 1:
                for i in range(0, image_embeds.shape[0]):
                    image_vit_features_path = os.path.join(vit_output_dir, f"output_image_vit_feats_{idx}_{i}.pt")
                    torch.save(image_embeds[i].unsqueeze(0), os.path.join(vit_output_dir, f"output_image_embeds_{idx}_{i}.pt"))
                    torch.save(image_vit_feats[i].unsqueeze(0), os.path.join( vit_output_dir, f"output_image_vit_feats_{idx}_{i}.pt"))
                    image_vit_features_paths[idx].append(image_vit_features_path)
            res_paths = save_and_concat_images(
                idx,
                image_vit_feats if image_vit_feats is not None else None,
                images_dir,
                label_file,
                aligner_net,
                vae_ref,
                device
            )

            result = {
                "sample_idx": idx,          
                "input_text":  decoded_input,
                "output_text": response,
                **res_paths,                    
            }

            append_jsonl(result, jsonl_path)
            idx += 1
            
            
        # if idx > 10:
        #     break


def save_and_concat_images(
    sample_idx: int,
    image_embeds: Optional[torch.Tensor],
    images_dir: str,
    label_file: str,
    aligner_net,
    vae_ref,
    device: torch.device,
    dataset_root: str = "/path/to/reasoning_maze"
) -> Dict[str, Any]:
    os.makedirs(images_dir, exist_ok=True)
    with open(label_file, "r", encoding="utf-8") as lf:
        labels = json.load(lf)

    # ---------- Input images ----------
    input_image_paths: List[str] = []
    if sample_idx < len(labels):
        for j, rel_raw in enumerate(labels[sample_idx].get("input_img", [])):
            raw_img_path = os.path.join(dataset_root, rel_raw)
            img_name = f"input-sample-{sample_idx}_img-{j}.png"
            save_path = os.path.join(images_dir, img_name)

            with Image.open(raw_img_path) as im:
                im.resize((224, 224), Image.LANCZOS).save(save_path)
            input_image_paths.append(img_name)           

    # ---------- Output images ----------
    output_image_paths: List[str] = []
    if image_embeds is not None and image_embeds.ndim >= 2:
        for j in range(image_embeds.size(0)):
            feat = image_embeds[j].to(device)          # [1, ...]
            decoded = decode_img(feat, aligner_net, vae_ref, device)  
            pil_img = transforms.ToPILImage()(decoded)

            img_name = f"sample-{sample_idx}_img-{j}.png"
            pil_img.save(os.path.join(images_dir, img_name))
            output_image_paths.append(img_name)

    # ---------- Concatenate ----------
    concat_img_name: Optional[str] = None
    combined = input_image_paths + output_image_paths
    if combined:
        pil_imgs = [Image.open(os.path.join(images_dir, p)) for p in combined]
        heights = [h for _, h in (im.size for im in pil_imgs)]
        min_h = min(heights)

        # (min_h × min_h)
        pil_imgs = [
            im if im.size == (min_h, min_h)
            else im.resize((min_h, min_h), Image.LANCZOS)
            for im in pil_imgs
        ]
        total_w = min_h * len(pil_imgs)
        mode = "RGBA" if any(im.mode == "RGBA" for im in pil_imgs) else "RGB"
        cat = Image.new(
            mode,
            (total_w, min_h),
            (255, 255, 255, 0) if mode == "RGBA" else (255, 255, 255),
        )
        x_off = 0
        for im in pil_imgs:
            cat.paste(im, (x_off, 0))
            x_off += min_h

        concat_img_name = f"concatenated-{sample_idx}.png"
        cat.save(os.path.join(images_dir, concat_img_name))

    # ---------- Assemble result ----------
    return {
        "input_image_paths":  input_image_paths,
        "output_image_paths": output_image_paths,
        "concat_img_path":    concat_img_name,
        "label_actions":      labels[sample_idx].get("label_actions", []) if sample_idx < len(labels) else [],
    }


###################
# Main Evaluation #
###################
def main():
    parser = argparse.ArgumentParser(description="Multimodal Model Evaluation using Trainer")
    parser.add_argument("--model_path", type=str, default="Junfeng5/Liquid_V1_7B", help="Path to the model checkpoint")
    parser.add_argument(
        "--decoder_path",
        type=str,
        required=True,
        help="Path of the pretrained Sketch Decoder.",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Output directory for evaluation results")
    parser.add_argument("--image_folder", type=str, default="generated_images", help="Folder name under output_dir to save generated images")
    parser.add_argument("--json_output_file", type=str, default="generated_outputs.json", help="JSON file (in output_dir) to store generated outputs")
    parser.add_argument("--device", type=str, default="cuda", help="Device for evaluation ('cuda' or 'cpu')")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--text-only', action="store_true", default=False, help="text-only baseline")
    parser.add_argument('--eval-only', action="store_true", default=False, help="evaluation only")
    
    # Add other parameters (e.g., model or dataset paths) as needed.
    # --- NEW ARGUMENTS FOR SHARDING ---
    parser.add_argument('--num-shards', type=int, default=1, help="Total number of shards to split the data into.")
    parser.add_argument('--shard-id', type=int, default=0, help="The ID of the shard to process (0-indexed).")

    # Add other parameters (e.g., model or dataset paths) as needed.
    args = parser.parse_args()

    # --- SHARDING LOGIC START ---
    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            raise ValueError(f"Shard ID must be between 0 and {args.num_shards - 1}, but got {args.shard_id}")

        # Modify the base output directory to be shard-specific to avoid file conflicts
        args.output_dir = os.path.join(args.output_dir, f"shard_{args.shard_id}")
        
        print(f"Loading full dataset from {args.data_path} for sharding...")
        with open(args.data_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        
        # Calculate shard boundaries
        total_size = len(full_data)
        # Use ceiling division to ensure all samples are covered
        shard_size = (total_size + args.num_shards - 1) // args.num_shards
        start_index = args.shard_id * shard_size
        end_index = min((args.shard_id + 1) * shard_size, total_size)
        
        sharded_data = full_data[start_index:end_index]

        # Define a path for the temporary sharded data file within the original output directory
        base_output_dir = os.path.dirname(args.output_dir) # e.g., 'evaluation_results'
        os.makedirs(base_output_dir, exist_ok=True)
        sharded_data_path = os.path.join(base_output_dir, f"data_shard_{args.shard_id}_of_{args.num_shards}.json")
        
        print(f"Shard {args.shard_id}/{args.num_shards}: Processing {len(sharded_data)} samples (from original index {start_index} to {end_index}).")
        print(f"Sharded data will be temporarily saved to: {sharded_data_path}")
        print(f"All outputs for this shard will be saved in: {args.output_dir}")

        with open(sharded_data_path, "w", encoding="utf-8") as f:
            json.dump(sharded_data, f)
            
        # Point args.data_path to the new sharded file for the rest of the script
        args.data_path = sharded_data_path
    # --- SHARDING LOGIC END ---
    
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    # # Construct the full path for the JSON output file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    json_output_path = os.path.join(args.output_dir, args.json_output_file)
    images_dir = os.path.join(args.output_dir, args.image_folder)
    vit_feature_dir = os.path.join(args.output_dir, "vit_features")
    if not os.path.exists(vit_feature_dir):
        os.makedirs(vit_feature_dir)

    if not args.eval_only:
        if "gemma" in args.model_path.lower(): 
            model_name = "gemma"   
            if "gemma-3-12b-it" in args.model_path.lower():
                model = Gemma3ForConditionalGeneration.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,  # or use fp16 if bf16 is unsupported
                    attn_implementation="flash_attention_2",
                ).to('cuda')
                is_original_model = True
            else:
                if os.path.exists(args.model_path):
                    shutil.copy('/path/to/gemma-3-12b-it/preprocessor_config.json', args.model_path)
        
                model = GemmaGenForConditionalGeneration.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,  # or use fp16 if bf16 is unsupported
                    attn_implementation="flash_attention_2",
                    #attn_implementation="eager",
                ).to('cuda')
                is_original_model = False
            untie_embeddings(model)
        elif "qwen" in args.model_path.lower():
            model_name = "qwen"
            if "qwen2.5-vl-7b" in args.model_path.lower():
                from transformers import Qwen2_5_VLForConditionalGeneration
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,  # or use fp16 if bf16 is unsupported
                    attn_implementation="flash_attention_2",
                ).to('cuda')
                is_original_model = True
            else:
                is_original_model = False
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
        test_dataset = MultimodalEvalDataset(tokenizer, processor, args.data_path, image_dir="/path/to/reasoning_maze/", is_original_model=is_original_model, model_name=model_name)   
        print(f"Loaded {len(test_dataset)} samples from {args.data_path}")
        
        run_evaluation(
            model=model,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            images_dir=images_dir,
            jsonl_path=json_output_path,
            label_file=args.data_path, 
            aligner_net=aligner_net, 
            vae_ref=vae_ref, 
            device=model.device,
            batch_size=args.batch_size,
            max_new_tokens=2500,
            vit_output_dir=vit_feature_dir,
            is_original_model=is_original_model,
            model_name=model_name
        )
        

    #####################################
    # Evaluate Generated Texts and Images Separately
    #####################################
    text_evaluation(json_output_path, text_evaluator, args.output_dir, args.data_path)
    if not args.text_only:
        image_evaluation(json_output_path, image_evaluator, args.output_dir, args.data_path, images_dir)
    

if __name__ == "__main__":
    main()