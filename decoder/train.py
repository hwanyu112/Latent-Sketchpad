import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from diffusers.models import AutoencoderKL
import open_clip
from torchvision import transforms
from PIL import Image
import os, sys, json
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities import grad_norm
from argparse import ArgumentParser
sys.path.append(os.path.abspath(__file__))
from vae_encoder import VaeEncoder  # Make sure to replace this with the actual path to your Encoder class definition
from data.quickdraw import QuickDraw, Category
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'aligner'))
from torchdata.stateful_dataloader import StatefulDataLoader
from vision_encoder_wrapper import VisionTransformerWrapper

import numpy, random

random.seed(42)
numpy.random.seed(42)
torch.set_float32_matmul_precision('high')

# Custom Dataset Class
class QuickDrawDualTransformDataset(QuickDraw):
    def __init__(self, root, transform_vit=None, transform_vae=None, download=True, gray=True, cate_num = -1, number_per_class=10000, image_size=224):
        self.data = QuickDraw(root=root, max_items_per_class=number_per_class, categories= list(Category)[:cate_num] if cate_num != -1 else list(Category), download=download, gray=gray, image_size=image_size)
        self.transform_vit = transform_vit
        self.transform_vae = transform_vae
        self.image_size = image_size

    def __getitem__(self, index):
        try:
            stroke_img, target, background_color = self.data[index]
            stroke_img = stroke_img.convert('RGB')
            backup_idx = index
            while stroke_img.size[0] == 1 or stroke_img.size[1] == 1:
                backup_idx = (backup_idx + 1) % len(self.data)
                stroke_img, target, background_color = self.data[backup_idx]
                stroke_img = stroke_img.convert('RGB')
            img_vit = self.transform_vit(stroke_img)
            img_vae = self.transform_vae(stroke_img)

            background_value = torch.tensor([-1.0, -1.0, -1.0] if background_color == 'black' else [1.0, 1.0, 1.0], dtype=torch.float32)
            background_value = background_value.view(3, 1, 1)

            mask = (img_vae != background_value).any(dim=0).float()
            mask = mask.unsqueeze(0).repeat(3, 1, 1)
            return img_vit, img_vae, mask
        except Exception as e:
            # Log the error if necessary
            print(f"Error in __getitem__ at index {index}: {e}")
            image_size = self.image_size
            empty_image = Image.new('RGB', (image_size, image_size), color=(0, 0, 0))  # Create a black image
            # Return default tensors to keep training going
            #default_img_transformed = torch.zeros((3, image_size, image_size))  # Assuming default size (224x224) for transformed images
            #default_img_normed = torch.zeros((3, image_size, image_size))       # Adjust size as needed
            #default_mask = torch.zeros((3, image_size, image_size))             # Adjust size as needed
            default_img_transformed = self.transform_vit(empty_image)
            default_img_normed = self.transform_vae(empty_image)
            default_mask = torch.zeros((3, image_size, image_size))             # Adjust size as needed
            return default_img_transformed, default_img_normed, default_mask

    def __len__(self):
        return len(self.data)

def focal_loss(input, target, foreground_mask, threshold = 0.1, focal_weight = 2):
    abs_diff = torch.abs(input - target)
    
    # Identify correct predictions (positive predictions)
    positive_mask = (abs_diff < threshold).float()  # Positive if abs difference is below threshold
    false_mask = 1 - positive_mask  # False predictions

    # Compute pixel-wise L2 loss
    l2_loss = F.mse_loss(input, target, reduction='none')
    """ print('original mse:', l2_loss.sum())
    print('number of false predictions:', false_mask.sum())
    print('foreground number:', foreground_mask.sum()) """

    
    # Apply focal weighting: reduce weight for positive predictions
    weights = positive_mask + false_mask * focal_weight

    # Apply weights to the loss
    weighted_loss = weights * l2_loss
    
    return weighted_loss.sum() / foreground_mask.sum() if foreground_mask.sum() > 1 else weighted_loss.mean()

# Reconstruct Image OOD Custom Step with W&B logging
def reconstruct_image_OOD(image_path, image_size, model, vision_encoder, vae_ref, device, logger, step):
    input_image = Image.open(image_path).convert("RGB")
    transform_vae = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = vision_encoder.image_transform(input_image).unsqueeze(0).to(device)
    vae_input = transform_vae(input_image).to(device)
    background_mask = (vae_input > 0).float()
    foreground_mask = 1 - background_mask
    with torch.no_grad():
        img_tokens = vision_encoder.encode_image(input_tensor).to(device)
        padding_mask = torch.zeros((img_tokens.size(0), img_tokens.size(1)), dtype=torch.bool).to(device)
        _, latent_output = model.encode(img_tokens, padding_mask)
        latent_tensor = latent_output.latent_dist.mode()
        decoded = vae_ref.decode(latent_tensor).sample
    
    reconstruction_loss = focal_loss(decoded.clamp(-1, 1), vae_input.unsqueeze(0), foreground_mask)

    reconstructed_image_tensor = decoded.squeeze(0)
    reconstructed_image_tensor = (reconstructed_image_tensor * 0.5 + 0.5).clamp(0, 1)
    reconstructed_image = transforms.ToPILImage()(reconstructed_image_tensor)

    original = vae_input.squeeze(0) * 0.5 + 0.5
    original = transforms.ToPILImage()(original)
    logger.log_metrics({'test_loss_decoded': reconstruction_loss}, step=step)

    if step == 0:
        logger.log_image(key="inference_original_image", images = [original], step = step)
    logger.log_image(key="inference_reconstructed_image", images = [reconstructed_image], step = step)


# Custom Callback for checkpointing and running custom step
class CustomCheckpointCallback(ModelCheckpoint):
    def __init__(self, every_n_train_steps, logger, *args, **kwargs):
        super().__init__(every_n_train_steps=every_n_train_steps, *args, **kwargs)
        self.logger = logger
        self.every_n_train_steps = every_n_train_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if trainer.global_step % self.every_n_train_steps == 0:
            reconstruct_image_OOD('test-abstract/maze.png', pl_module.image_size, pl_module.aligner_net, pl_module.vision_encoder, pl_module.vae_ref, pl_module.device, self.logger, trainer.global_step)


# Define LightningModule with W&B Logging
class LitAlignerModel(L.LightningModule):
    def __init__(self, aligner_net, vision_encoder, vae_encoder, vae_ref, learning_rate, need_dense, non_background_weight, image_size):
        super().__init__()
        self.aligner_net = aligner_net
        self.vision_encoder = vision_encoder
        self.vae_encoder = vae_encoder
        self.vae_ref = vae_ref
        self.learning_rate = learning_rate
        self.need_dense = need_dense
        self.non_background_weight = non_background_weight
        self.validation_epoch_outputs = []
        self.image_size = image_size


    def forward(self, img_tokens, padding_mask, vae_embedding):
        return self.aligner_net(img_tokens, padding_mask, vae_embedding)

    def step(self, batch, batch_idx):
        imgs_vit, imgs_vae, foreground_masks = batch
        imgs_vit = imgs_vit.to(self.device)
        imgs_vae = imgs_vae.to(self.device)

        foreground_masks = foreground_masks.to(self.device)

        img_tokens = self.vision_encoder.encode_image(imgs_vit)
        padding_mask = torch.zeros((img_tokens.size(0), img_tokens.size(1)), dtype=torch.bool).to(self.device)
        target_latents, vae_embedding = self.vae_encoder(imgs_vae, self.need_dense)
        res = self.aligner_net(img_tokens, padding_mask, vae_embedding)

        latent_output = res['output']
        latent_tensor = latent_output.latent_dist.sample()
        decoded = self.vae_ref.decode(latent_tensor).sample

        """ image_reconstruction_loss = F.mse_loss(decoded, imgs_vae, reduction='none')
        weighted_loss = image_reconstruction_loss * (1 + (self.non_background_weight - 1) * foreground_masks) """
        reconstruction_loss = focal_loss(decoded, imgs_vae, foreground_masks)
        latent_loss =  torch.clamp(F.gaussian_nll_loss(latent_output.latent_dist.mean, 
                                target_latents.sample().to(self.device), 
                                torch.clamp(latent_output.latent_dist.std**2, min=1e-6).to(self.device), 
                                eps=1e-5,
                                reduction='mean'),
                                0)
        if self.need_dense:
            #latent_loss = F.mse_loss(latent_output.latent_dist.mean, target_latents.mean.to(self.device)) +\
            #            F.mse_loss(latent_output.latent_dist.std, target_latents.std.to(self.device))
            latent_loss += F.mse_loss(latent_output.latent_dist.std, target_latents.std.to(self.device))
        return reconstruction_loss, latent_loss, res['mse_loss']

    def training_step(self, batch, batch_idx):
        weighted_loss, latent_loss, embed_loss = self.step(batch, batch_idx)
        loss = weighted_loss + embed_loss + latent_loss
        self.log_dict({'train_loss': loss, 
                       'latent_loss': latent_loss, 
                       'embed_loss': embed_loss,
                       'image_loss': weighted_loss},
                       logger=True)

        return loss

    def on_train_start(self):
        # Move models to the correct device using self.device
        self.vae_ref.to(self.device)
        self.vae_encoder.to(self.device)
        self.aligner_net.to(self.device)
        self.vision_encoder.move_to(self.device)

    def validation_step(self, batch, batch_idx):
        if self.trainer.sanity_checking:
            # Move models to the correct device using self.device
            self.vae_ref.to(self.device)
            self.vae_encoder.to(self.device)
            self.aligner_net.to(self.device)
            self.vision_encoder.move_to(self.device)
        weighted_loss, latent_loss, embed_loss = self.step(batch, batch_idx)
        loss = weighted_loss + embed_loss + latent_loss
        self.validation_epoch_outputs.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_outputs):
            # Aggregate results for the entire validation epoch
            avg_loss = torch.stack(self.validation_epoch_outputs).mean()
            self.log('val_loss', avg_loss, logger=True, sync_dist=True)
            # Clear the list for the next epoch
            self.validation_epoch_outputs.clear()

    def configure_optimizers(self):
        return optim.Adam(self.aligner_net.parameters(), lr=self.learning_rate)

    def state_dict(self):
        # Only save weights being trained
        #return {k: v for k, v in self.aligner_net.state_dict().items() if v.requires_grad}
        return self.aligner_net.state_dict()
    def load_state_dict(self, state_dict, strict):
        self.aligner_net.load_state_dict(state_dict, strict)
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        super().on_before_optimizer_step(optimizer)
        norms = grad_norm(self, norm_type=2)
        self.log('grad_norm_total', norms['grad_2.0_norm_total'], logger=True)

# Prepare DataLoader
def prepare_dataloaders(batch_size, train_ratio, val_ratio, num_workers, transform_vit, transform_vae, gray, cate_num, number_per_class, image_size):
    dataset = QuickDrawDualTransformDataset(root=os.getenv('DATA_DIR', '.'), transform_vit=transform_vit, transform_vae=transform_vae, download=True, gray=gray, cate_num=cate_num, number_per_class=number_per_class, image_size=image_size)
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f'train_size:{train_size}, val_size:{val_size}, test_size:{test_size}')
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        numpy.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = StatefulDataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=False, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g,
                            drop_last=True)
    val_loader = StatefulDataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=False, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g)
    test_loader = StatefulDataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            pin_memory=False, num_workers=num_workers,
                            worker_init_fn=seed_worker, generator=g)

    
    return train_loader, val_loader, test_loader

# Main Training Function
def main(args):
    seed_everything(42, workers=True)

    # Configuration
    default_config = {
        'project_name': 'clip-sdxl-vae-color-transformer-scratch',
        "learning_rate": 1e-4,
        "epochs":100,
        "batch_size": 64,
        "clip_model": "ViT-L-14",
        'non_background_weight': 1,
        'setting': 'black_white_background_mask_weighted_mse',
        'checkpoints_dir': 'checkpoints',
        'restore_optimizer': False,
        'dataset':'quickdraw-344-classes',
        'cate_num': 5,
        'number_per_class': 10000,
        'train_split': 0.99,
        'val_split': 0.003,
        'dense_align': True,
        "gray_image": True,
        'eval_every_n_steps': 500,
        'vision_model_name': 'qwen',
        'image_size': 224,
        'layer': 12,
        'input_dim': 1664,
        "accumulate_grad_batches": 4,
        "causal_mask": False,
    }
    # Check for environment variable for config file
    config_file_path = os.getenv('CONFIG_FILE_PATH', 'configs/dense-config-scratch.json')

    if config_file_path and os.path.exists(config_file_path) and config_file_path.endswith('.json'):
        with open(config_file_path, 'r') as f:
            config_from_file = json.load(f)
        default_config.update(config_from_file)

    batch_size = default_config['batch_size']
    learning_rate = default_config['learning_rate']
    num_epochs = default_config['epochs']
    checkpoint_dir = default_config['checkpoints_dir']
    mount_root = os.getenv('MOUNT_DIR', '')
    image_size = default_config['image_size']
    grid_size = int(image_size / 8)
    vision_model_name = default_config['vision_model_name']
    project_name = f"{vision_model_name.split('/')[0]}-{default_config['project_name']}"
    accumulate_grad_batches = default_config['accumulate_grad_batches']

    # Initialize W&B logger for PyTorch Lightning
    wandb_logger = WandbLogger(project=project_name, config=default_config)

    # Initialize VAE and ViT models
    vit_feature_only = os.environ.get('VIT_FEATURES_ONLY', '0') == '1'
    vision_encoder = VisionTransformerWrapper(vision_model_name, mount_root, image_size, vit_feature_only)
    vae_ref = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    vae_ref.eval()
    for param in vae_ref.parameters():
        param.requires_grad = False
    vae_encoder = VaeEncoder(
        in_channels=vae_ref.config.in_channels,
        out_channels=vae_ref.config.latent_channels,  # corresponds to latent_channels in VAE
        down_block_types=vae_ref.config.down_block_types,
        block_out_channels=vae_ref.config.block_out_channels,
        layers_per_block=vae_ref.config.layers_per_block,
        act_fn=vae_ref.config.act_fn,
        norm_num_groups=vae_ref.config.norm_num_groups,
        double_z=True,  # As per the VAE initialization, this is set to True
        mid_block_add_attention=vae_ref.config.mid_block_add_attention  # Ensure this matches the VAE setting
    )
    vae_encoder.load_state_dict(torch.load(os.path.join(mount_root, 'vae-encoder.pth')))
    vae_encoder.eval()
    for param in vae_encoder.parameters():
        param.requires_grad = False

    upload_dir = os.path.join(mount_root, project_name, checkpoint_dir)
    os.makedirs(upload_dir, exist_ok=True)

    need_dense = default_config['dense_align']
    if need_dense:
        from dense_aligner import ClipToLatentAligner # Import your aligner class
    else:
        from gaussian_aligner import ClipToLatentAligner
    # Initialize your aligner model and custom encoder
    aligner_net = ClipToLatentAligner(vae_encoder, default_config['input_dim'], 512, grid_size, default_config['layer'], default_config['causal_mask'])

    # Initialize LightningModule
    model = LitAlignerModel(aligner_net, vision_encoder, vae_encoder, vae_ref, learning_rate, default_config['dense_align'], default_config['non_background_weight'], image_size)
    # Define Custom Checkpoint Callback
    checkpoint_callback = CustomCheckpointCallback(
        every_n_train_steps=default_config['eval_every_n_steps'],
        logger=wandb_logger,
        dirpath=upload_dir,
        filename='clip-vae_aligner-{epoch:02d}-{step}-{val_loss:.2f}',
        #save_top_k=-1,
        #save_last='link',
        save_top_k=10,              # keep only top 10 checkpoints
        save_last='link',
        monitor='val_loss',         # metric to monitor
        mode='min', 
    )

    # Trainer setup
    trainer = Trainer(
        max_epochs=num_epochs,
        devices=args.gpus,
        num_nodes=args.nodes,
        logger=wandb_logger,  # Use the W&B logger
        callbacks=[checkpoint_callback],
        val_check_interval=default_config['eval_every_n_steps'] * accumulate_grad_batches,
        log_every_n_steps=default_config['eval_every_n_steps'],
        accelerator="gpu",
        strategy="ddp",
        deterministic=True,
        #precision='bf16-mixed',
        gradient_clip_val=1.0,
        use_distributed_sampler=True,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    # Prepare DataLoader
    # Define transformations for CLIP and VAE
    transform_vae = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to match CLIP's preprocessing
    ])
    train_loader, val_loader, test_loader = prepare_dataloaders(batch_size, default_config['train_split'], default_config['val_split'], 
                                                                num_workers=0,
                                                                transform_vit=vision_encoder.image_transform,
                                                                transform_vae=transform_vae, 
                                                                gray=default_config['gray_image'],
                                                                cate_num=default_config['cate_num'],
                                                                number_per_class=default_config['number_per_class'],
                                                                image_size=image_size)
    # Train model
    trainer.fit(model, train_loader, val_loader, ckpt_path='last')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    args = parser.parse_args()

    main(args)