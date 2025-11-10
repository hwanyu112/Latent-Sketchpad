export GENERATION_TYPE=agent
python 4o_ls.py --model_path /path/to/latent_sketchpad \
                --decoder_path /path/to/sketch_decoder_qwen25_vl.ckpt  \
                --data_path /path/to/data.json \
                --image_folder ls_imgs/ \
                --output_dir /path/to/output