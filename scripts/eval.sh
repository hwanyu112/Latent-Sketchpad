export GENERATION_TYPE=multimodal # replace with 'text_only' for text-only generation
python evaluate.py --model_path /path/to/model \
                --decoder_path /path/to/sketch_decoder.ckpt  \
                --data_path /path/to/test_data.json \
                --image_folder imgs/ \
                --output_dir /path/to/output_dir
                
                
