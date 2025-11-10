# bash scripts/patch_import_utils.sh ### The enviroment for training Qwen2.5-VL may need this patch_import_utils.sh
train.py --data_path /path/to/reasoning_maze/interleave_sft_data.json  \
      --decoder_path /path/to/sketch_decoder.ckpt \
      --image_dir /path/to/reasoning_maze \
      --model_path path_or_repo_id/of/base_model \
      --learning_rate 1e-4 --max_grad_norm 1.0 --num_train_epochs 2 \
      --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --per_device_eval_batch_size 1 --weight_decay 0.01 \
      --ds_config ./ds_cfg.json \
      --save_steps 200 --eval_steps 100 --logging_steps 100 --resume_from_checkpoint True \
      --output_dir /path/to/output_dir/ --wandb_project project_name \
      --validation_split 0.005 --augment --image_loss_weight 0.0 --unfreeze-connector --stage1 \
      --sum-loss --loss_type l1