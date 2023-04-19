import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils import plot_images, save_images, setup_logging, get_data
from modules import UNet_conditional, EMA
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train_model(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("/content/drive/MyDrive/Conquer_Diffusion/Diff/results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join("/content/drive/MyDrive/Conquer_Diffusion/Diff/results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join("/content/drive/MyDrive/Conquer_Diffusion/Diff/model", args.run_name, f"ckpt_{epoch}_.pt"))
            torch.save(ema_model.state_dict(), os.path.join("/content/drive/MyDrive/Conquer_Diffusion/Diff/model", args.run_name, f"ema_ckpt_{epoch}_.pt"))
            torch.save(optimizer.state_dict(), os.path.join("/content/drive/MyDrive/Conquer_Diffusion/Diff/model", args.run_name, f"optim_{epoch}_.pt"))



if __name__ == "__main__":
    
    # ARGUMENTS PARSER
    p = argparse.ArgumentParser()
    
    
    p.add_argument("--run_name", type=str, default="DDPM_conditional", help='(글자)임의의 러닝 폴더 이름을 넣어주세요')
    p.add_argument("--epochs", type=int, default=300, help='(정수) 훈련 횟수를 정해주세요')
    p.add_argument("--batch_size", type=int, default=8, help='(정수)배치 사이즈를 정해주세요')
    p.add_argument("--image_size", type=int, default=64, help='(정수)이미지 사이지를 정해주세요')
    p.add_argument("--num_classes", type=int, default=10, help='(정수)클래스의 갯수를 넣어주세요')
    p.add_argument("--dataset_path", type=str, default=r"/content/Diffusion-Models-pytorch-module/datasets/Landscape_classifier_02/training", help='(글자)데이터셋 경로를 넣어주세요')
    p.add_argument("--device", type=str, default="cuda", help='(글자)GPU 로 돌릴지 CPU로 돌릴지 정해주세요!')
    p.add_argument("--lr", type=float, default=3e-4, help='(부동소수점)러닝레이트를 넣어주세요')
                    
    args = p.parse_args()
                   
    train_model(args)
