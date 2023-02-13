import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import isolate_rng
import torch
import torchvision
import subprocess
from lora_diffusion import patch_pipe
import os
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import tempfile
import numpy as np
import sys
import traceback
from torchvision import transforms

class OutputMonitor(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        x, _ = batch
        z = None
        for i in range(x.shape[1]):
            encoded = pl_module.encoder(x[:, i, ...])
            if z is None:
                z = encoded
            else:
                z = torch.cat((z, encoded), dim=1)
        y = pl_module.lookup(z.unsqueeze(1)).squeeze(1)
        trainer.logger.experiment.add_histogram("lookup_histogram", y, global_step=trainer.global_step)
        trainer.logger.experiment.add_histogram("encoder_histogram", z, global_step=trainer.global_step)
        
        z2 = pl_module.encoder(torch.zeros_like(data).uniform_(-1, 1))
        y2 = pl_module.lookup(z2.unsqueeze(1)).squeeze(1)
        pl_module.log("lookup_diff_from_noise", abs(y2 - y).mean())
                   
class InputMonitor(pl.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            trainer.logger.experiment.add_histogram("image_histogram", x, global_step=trainer.global_step)
            trainer.logger.experiment.add_histogram("embeds_histogram", y, global_step=trainer.global_step)
            trainer.logger.experiment.add_histogram("embeds_normalized_histogram", pl_module.embed_normalizer(y), global_step=trainer.global_step)

class GenerateCallback(pl.Callback):
    def _get_train_images_part(self, data_loader):
        for img in data_loader:
            return img
            
    def __init__(self, data_loader, every_n_epochs = 5):
        super().__init__()
        self.input_imgs = self._get_train_images_part(data_loader)
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                z = pl_module.encoder(input_imgs)
                reconst_imgs = pl_module.decoder(z)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)

class GenerateFromLoraCallback(pl.Callback):
    def __init__(self, test_images_path, every_n_epochs = 5):
        super().__init__()
        self.test_images_path = test_images_path
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def _inner(self, trainer, pl_module):
        try:
            if trainer.current_epoch % self.every_n_epochs == 0:
                file_path = os.path.abspath(os.path.dirname(__file__))
                cli_path = os.path.abspath(os.path.join(file_path, "..", "bin", "leap_lora"))
                checkpoints_path = f"{trainer.logger.log_dir}/checkpoints"
                if not os.path.exists(checkpoints_path):
                    print("Rejecting GenerateFromLoraCallback callback as there's no checkpoints yet")
                    return
                checkpoint_path = os.path.join(checkpoints_path, os.listdir(checkpoints_path)[0])
                model_id = "stabilityai/stable-diffusion-2-1-base"
                print("checkpoint_path", checkpoint_path)
                with tempfile.TemporaryDirectory() as tmpdirname:
                    shell_command = f"""{sys.executable} -u {cli_path} \
                        --pretrained_model_name_or_path="{model_id}"  \
                        --instance_data_dir="{self.test_images_path}" \
                        --leap_model_path={checkpoint_path} \
                        --output_dir="{tmpdirname}" \
                        --train_text_encoder \
                        --resolution=512 \
                        --train_batch_size=1 \
                        --gradient_accumulation_steps=4 \
                        --scale_lr \
                        --learning_rate_unet=1e-4 \
                        --learning_rate_text=1e-5 \
                        --learning_rate_ti=2e-3 \
                        --color_jitter \
                        --lr_scheduler="constant" \
                        --lr_scheduler_lora="constant" \
                        --lr_warmup_steps=10 \
                        --placeholder_tokens="<s1>" \
                        --use_template="object" \
                        --save_steps=100 \
                        --max_train_steps_ti=100 \
                        --max_train_steps_tuning=100 \
                        --perform_inversion=True \
                        --clip_ti_decay \
                        --weight_decay_ti=0.000 \
                        --weight_decay_lora=0.001 \
                        --continue_inversion \
                        --continue_inversion_lr=1e-4 \
                        --device="cuda:0" \
                        --lora_rank=1
                    """.strip()
                    p = subprocess.Popen(shell_command, shell=True)
                    p.communicate()
                    new_tensors_path = os.path.join(tmpdirname, "step_100.safetensors")
                    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
                        "cuda"
                    )
                    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                    patch_pipe(
                        pipe,
                        new_tensors_path,
                        patch_text=True,
                        patch_ti=True,
                        patch_unet=True,
                    )
                    with isolate_rng():
                        image = pipe("<s1>", num_inference_steps=25, guidance_scale=9).images[0]
                        image = transforms.ToTensor()(image)
                        trainer.logger.experiment.add_image("Test gen", image, global_step=trainer.global_step)
        except:
            print("Error with GenerateFromLoraCallback!")
            traceback.print_exception(*sys.exc_info()) 

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        self._inner(trainer, pl_module)
        torch.cuda.empty_cache()