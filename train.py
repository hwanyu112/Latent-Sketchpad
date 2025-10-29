import argparse
import os
from numpy import ndarray
import torch
import shutil
import torch.distributed as dist
from torch.utils.data import Dataset, random_split
from torch.functional import F
from tqdm import tqdm
import wandb  # ensure wandb is installed: pip install wandb
import uuid  # Used for generating a random run name

from transformers import AutoTokenizer, AutoProcessor, TrainingArguments, set_seed
from transformers import AutoConfig, AutoTokenizer, AddedToken
# from transformers.trainer_utils import get_last_checkpoint  # Updated import

# Import your custom model class. Adjust this import as necessary.
from multimodal_trainer import MultimodalTrainer
from model.uni_gemma import GemmaGenForConditionalGeneration
from model.uni_qwen import UniQwenForConditionalGeneration
from data.dataset import MultimodalDataset, DataCollatorForInputs
from gen_utils import untie_embeddings, get_last_checkpoint
import random

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gen_utils import left_padding

torch.autograd.set_detect_anomaly(True)

def count_trainable_params(model):
    """Return the number of parameters that will receive gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

###############################################################################
# Main training routine.
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on pre-tokenized data with multi-node support using torchrun and wandb logging."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/gemma-3-12b-it",
        help="Path or identifier of the pretrained model.",
    )
    parser.add_argument(
        "--decoder_path",
        type=str,
        required=True,
        help="Path of the pretrained Sketch Decoder.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        nargs="+",             
        help="Path(s) to the data file(s)."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the image directory."
    )
    parser.add_argument(
        "--ds_config",
        type=str,
        default="ds_cfg.json",
        help="Path to the deepspeed config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory where the fine-tuned model will be saved.",
    )
        # Validation split and evaluation steps.
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.02,
        help="Fraction of the dataset to use for evaluation (0.0 to 1.0).",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Frequency (in steps) for running evaluation.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=bool,
        default=False,
        help="Whether to resume training from the last checkpoint.",
    )
    parser.add_argument("--ood_query_path", type=str, default="ood_query.pt", help="Path to the ood query file.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device during training.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Frequency of logging steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Frequency of model checkpoint saves.")
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--update_embed', action="store_true", default = False, help="whether to update embed_token weights")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--image_seq_len", type=int, default=1024, help="Image sequence length per image.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps.")
    parser.add_argument('--weight_decay', type=float, default=0.1, help="Weight decay for AdamW.")
    parser.add_argument('--lr_scheduler', type=str, default="cosine", help="Learning rate scheduler.")
    # WandB-related arguments.
    parser.add_argument("--wandb_project", type=str, default="chameleon-self-reflection", help="WandB project name.")
    parser.add_argument('--text_embed', action="store_true", default = False, help="whether to update text embed weights")
    parser.add_argument('--image_embed', action="store_true", default = False, help="whether to update image embed weights")
    parser.add_argument('--discrepancy_weight', type=float, default = 0.0, help="discrepancy loss weight")
    parser.add_argument('--image_loss_weight', type=float, default = 1.0, help="image cross-entropy loss weight")
    parser.add_argument('--text_loss_weight', type=float, default = 1.0, help="text cross-entropy loss weight")
    parser.add_argument('--sum-loss', action="store_true", default=False, help="whether to sum up seperate text and image loss")
    parser.add_argument('--image-logits-only', action="store_true", default=False, help="whether to only fine-tune image logits")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4, help="Batch size per device during evaluation.")
    parser.add_argument('--eval_metric_for_best_model', type=str, default="eval_loss", help="Metric for best model.")
    parser.add_argument('--no-sample', action="store_true", default=False, help="whether to do turn off sampling during generation")
    parser.add_argument('--wo-image-logits', action="store_true", default=False, help="whether to freeze image logits in lm_head")
    parser.add_argument('--use-lora', action="store_true", default=False, help="whether to use lora")
    parser.add_argument('--lora_r', type=int, default=8, help="lora rank")
    parser.add_argument('--lora_alpha', type=int, default=16, help="lora alpha")
    parser.add_argument('--lora_dropout', type=float, default=0.1, help="lora dropout")
    parser.add_argument('--loss_type', type=str, default="mse", help="loss type")
    parser.add_argument('--freeze-backbone', action="store_true", default=False, help="whether to freeze backbone")
    parser.add_argument('--unfreeze-connector', action="store_true", default=False, help="whether to unfreeze connector")
    parser.add_argument('--augment', action="store_true", default=False, help="whether to augment the input image")
    parser.add_argument('--stage1', action="store_true", default=False, help="whether to use text-only data")
    # Parse known args so that any additional [other args] are captured.
    args, extra_args = parser.parse_known_args()
    if extra_args:
        print(f"Extra arguments provided: {extra_args}")
    
    torch.manual_seed(args.seed)
    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] = os.environ.get('LOCAL_RANK')
    global_rank = None
    # Setup for distributed training if --local_rank is specified.
    if args.local_rank is not None or local_rank != -1:
        if args.local_rank is None:
            args.local_rank = local_rank if local_rank != -1 else args.local_rank
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        print(f"[Rank {dist.get_rank()}] Distributed training initialized. World size: {dist.get_world_size()}")
        global_rank = dist.get_rank()
    else:
        print("Running in single-process mode.")

    # Set environment variables for wandb (only on the main process).
    if global_rank == 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        run_name = f"run-{uuid.uuid4().hex[:6]}"
        wandb.init(project=args.wandb_project, config=vars(args), name=run_name)
        print(f"Initialized wandb run with name: {run_name}")

    # Load the pretrained model on the appropriate device.
    if "gemma" in args.model_path.lower():
        model = GemmaGenForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,  # or use fp16 if bf16 is unsupported
            attn_implementation="flash_attention_2",
        ).to('cuda')
        untie_embeddings(model)
        model_name = "gemma"
    elif "qwen" in args.model_path.lower():
        model = UniQwenForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,  # or use fp16 if bf16 is unsupported
            attn_implementation="flash_attention_2",
        ).to('cuda')
        model_name = "qwen"

    image_token_index = torch.tensor(model.image_token_index, dtype=torch.int64, device='cuda')
 
    # Unfreeze all parameters first
    for param in model.parameters():
        param.requires_grad = True
    if not args.text_embed:
        # Freeze all parameters of embed_tokens
        if model_name == "gemma":
            for param in model.language_model.model.embed_tokens.parameters():
                param.requires_grad = False
        elif model_name == "qwen":
            for param in model.model.language_model.embed_tokens.parameters():
                param.requires_grad = False
    
    if model_name == "gemma":
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        if not args.unfreeze_connector:
            for param in model.multi_modal_projector.parameters():
                param.requires_grad = False
    elif model_name == "qwen":
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        if args.unfreeze_connector:
            for param in model.visual.merger.mlp.parameters():
                param.requires_grad = True
    
    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    # Unfreeze regression head
    for param in model.regression_head.parameters():
        param.requires_grad = True
    
    if "stage2" in args.output_dir:
        ignore_image = False
        if model_name == "qwen":
            shutil.copy('/path/to/Qwen2.5-VL-7B-Instruct/preprocessor_config.json', args.model_path)
        elif model_name == "gemma":
            shutil.copy('/path/to/gemma-3-12b-it/preprocessor_config.json', args.model_path)
    else:
        ignore_image = True

    # Apply LoRA fine-tuning if enabled; otherwise, use the original freezing strategy.
    if args.use_lora:
        target_keys = {"q_proj", "k_proj", "v_proj", "o_proj"}#, "gate_proj", "up_proj", "down_proj"}
        lora_target_modules = set() # Use a set to avoid duplicates

        for name, module in model.named_modules():
            # Check if the module name ends with one of our target keys
            if name.split('.')[-1] in target_keys:
                # CRITICAL: Exclude any module whose path includes 'visual'
                if 'visual' not in name and 'vision_tower' not in name and "regression_head" not in name:
                    lora_target_modules.add(name)

        # print(f"Identified LoRA target modules: {lora_target_modules}")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            # target_modules=['q_proj', "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],    # "gate_proj", "up_proj", "down_proj"
            target_modules=list(lora_target_modules),
            lora_dropout=args.lora_dropout,
            bias="none",
            modules_to_save=['lm_head', 'visual.merger.mlp'] if model_name == "qwen" else ['language_model.lm_head', 'multi_modal_projector'],
            task_type="CAUSAL_LM",

        )
        model = get_peft_model(model, lora_config)
        model = model.merge_and_unload()
        print("LoRA fine-tuning is enabled.")

    modules = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            # keep only first `depth` parts of the name
            high_level = ".".join(name.split(".")[:3])
            modules.add(high_level)
    for m in sorted(modules):
        print(m)
    n_trainable = count_trainable_params(model)
    print(f'\nTotal trainable parameters: {n_trainable:,} '
        f'(~{n_trainable * 2 / 1024 / 1024:.2f} MB in bf16)')

    # Load the tokenizer (needed for padding)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)

    ###########################################################################
    # Load and split the dataset into training and evaluation sets.
    ###########################################################################
    full_dataset = MultimodalDataset(
        tokenizer, processor, model, 
        json_file=args.data_path, 
        image_dir=args.image_dir, 
        checkpoint_path=args.decoder_path, 
        feature_dim=model.config.vision_config.hidden_size, 
        augment=args.augment, 
        stage1=args.stage1, 
        model_name=model_name,
        image_token_index=image_token_index,
        boi_id=model.config.boi_token_index if model_name == "gemma" else model.config.vision_start_token_id,
        eoi_id=model.config.eoi_token_index if model_name == "gemma" else model.config.vision_end_token_id,
        ignore_image=ignore_image
        )
    dataset_size = len(full_dataset)
    eval_size = int(dataset_size * args.validation_split)
    train_size = dataset_size - eval_size
    if train_size <= 0 or eval_size <= 0:
        raise ValueError("Invalid split: Adjust --validation_split so both splits have at least one example.")

    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    print(f"Total dataset size: {dataset_size} | Train: {train_size} | Eval: {eval_size}")

    # Define training arguments. Note the "report_to" parameter is set to ["wandb"].
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="steps",  # This ensures step-based checkpoint saving.
        save_steps=args.save_steps,
        save_total_limit=20,
        bf16=True,  # Set to True if your hardware supports bf16; otherwise, consider fp16=True.
        eval_strategy="steps",  # Enable evaluation.
        eval_steps=args.eval_steps,
        local_rank=args.local_rank,  # Enables distributed training.
        report_to=["wandb"],  # Enable wandb logging.
        # You could also process extra args here if needed.
        #fsdp="full_shard",
        deepspeed=args.ds_config,  # Path to DeepSpeed config file        
        gradient_checkpointing=True,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        metric_for_best_model=args.eval_metric_for_best_model,
        greater_is_better=False,
        load_best_model_at_end=True,
        seed=args.seed,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        batch_eval_metrics=True,
    )

    # Create a data collator to pad sequences in a batch.
    data_collator = DataCollatorForInputs(tokenizer.pad_token_id, model_name)

    ###########################################################################
    # Resume from checkpoint if it exists.
    ###########################################################################
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            print(f"Resuming training from checkpoint: {last_checkpoint}")
            training_args.load_weights_from = last_checkpoint

    # Initialize the Trainer.
    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        image_loss_weight=args.image_loss_weight,
        text_loss_weight=args.text_loss_weight,
        sum_loss=args.sum_loss,
        image_token_index=image_token_index,
        loss_type=args.loss_type,
        boi_id=model.config.boi_token_index if model_name == "gemma" else model.config.vision_start_token_id,
        eoi_id=model.config.eoi_token_index if model_name == "gemma" else model.config.vision_end_token_id,
    )
    # Start fine-tuning. If a checkpoint exists, resume training from that checkpoint.
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint and last_checkpoint)
    trainer.save_model()
    trainer.save_state()
    if global_rank == 0:
        print(f"Model saved to {args.output_dir}")
        wandb.finish()


if __name__ == "__main__":
    main()
