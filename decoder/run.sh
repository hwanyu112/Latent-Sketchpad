export CONFIG_FILE_PATH=configs/vit-only-qwen25-12-224-80G.json
export WANDB_API_KEY=YOUR_WANDB_API_KEY
export MOUNT_DIR=/path/to/visualizer
export DATA_DIR=/path/to/data
export HF_TOKEN=YOUR_HF_TOKEN
export VIT_FEATURES_ONLY=1
python train.py --nodes=1 --gpus=1
