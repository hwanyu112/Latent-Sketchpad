import torch
import os, sys, json
import torch
sys.path.append(os.path.abspath(__file__))
from visual import VisionTransformer
from transformers import AutoProcessor
import open_clip
from transformers import AutoModel, CLIPImageProcessor
from torchvision import transforms
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from qwen25_vision_encoder import Qwen2_5_VisionTransformer

class VisionTransformerWrapper:
    def __init__(self, model_name, model_dir, image_size, vit_feature_only = False):
        self.model_name = model_name
        self.clip_processor = None
        self.mm_projector = None
        self.image_size = image_size
        self.vision_encoder = self.get_vision_encoder()
        self.image_transform = self.get_processor()
        self.vit_feature_only = vit_feature_only
        print('vit_feature_only:', self.vit_feature_only)

    def move_to(self, device):
        self.vision_encoder.to(device)
        if self.mm_projector:
            self.mm_projector.to(device)

    def get_vision_encoder(self):
        if self.model_name.lower().startswith('openclip'):
            vision_encoder, _, image_processor = open_clip.create_model_and_transforms(self.model_name.split('/')[-1])
            vision_encoder.visual.output_tokens = True
            self.clip_processor = image_processor
        elif self.model_name.startswith('google/gemma-3'):
            model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-12b-it", low_cpu_mem_usage = True, torch_dtype=torch.bfloat16)
            vision_encoder = model.vision_tower
            self.mm_projector = model.multi_modal_projector
            self.mm_projector.eval()
            for param in self.mm_projector.parameters():
                param.requires_grad = False
            self.patches_per_image = int(model.config.vision_config.image_size // model.config.vision_config.patch_size)
        elif self.model_name.startswith('Qwen/Qwen2.5'):
            vision_encoder = Qwen2_5_VisionTransformer.from_pretrained("/path/to/visualizer/qwen2_5_visual")
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        print(f'loading vision encoder: {self.model_name}')
        vision_encoder.eval()
        for param in vision_encoder.parameters():
            param.requires_grad = False
        return vision_encoder

    def pool_vit_features(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.mm_projector.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)
        normed_vision_outputs = self.mm_projector.mm_soft_emb_norm(pooled_vision_outputs)
        return normed_vision_outputs.type_as(vision_outputs)

    def encode_image(self, input_tensor):
        if self.model_name.startswith('Qwen/Qwen2.5'):
            image_grid_thw = torch.tensor([[ input_tensor.size(0), 32, 32]]).to(self.vision_encoder.device)
            img_tokens = self.vision_encoder(input_tensor.to(torch.float16), image_grid_thw)
            img_tokens = img_tokens.reshape(input_tensor.size(0), -1, img_tokens.size(-1))
        elif self.model_name.lower().startswith('openclip'):
            img_tokens = self.vision_encoder.visual(input_tensor)[1]
        elif self.model_name.startswith('google/gemma-3'):
            image_outputs = self.vision_encoder(input_tensor.to(torch.float16)).last_hidden_state
            if not self.vit_feature_only:
                img_tokens = self.mm_projector(image_outputs).to(torch.float32)
            else:
                img_tokens = self.pool_vit_features(image_outputs).to(torch.float32)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
        return img_tokens
    
    def get_processor(self):
        if self.model_name.startswith('Qwen/Qwen2.5'):
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            import torchvision.transforms.functional as F
            from PIL import Image
            return lambda images: processor.image_processor(
                F.resize(images if isinstance(images, Image.Image) else Image.fromarray(images),(448, 448)),
                return_tensors='pt', 
                do_resize=False
                )['pixel_values'].squeeze(0)
        elif self.model_name.lower().startswith('openclip'):
            #return self.clip_processor
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif self.model_name.startswith('google/gemma-3'):
            processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it").image_processor
            images_kwargs = {
            "do_pan_and_scan": False,
            "pan_and_scan_min_crop_size": 256,
            "pan_and_scan_max_num_crops": 4,
            "pan_and_scan_min_ratio_to_activate": 1.2,
            }
            return lambda images: processor(images=images, return_tensors='pt', **images_kwargs)['pixel_values'].squeeze(0)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")


