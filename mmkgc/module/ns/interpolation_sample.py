import os

import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torch.nn import functional as F

from mmkgc.module.ns.Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer


class InterpolationSampler:
    def __init__(self, ent_tot,
                 in_path,
                 diffusion_model=None,
                 beta_1=1e-4,
                 beta_T=0.01,
                 T=100):
        self.ent_tot = ent_tot
        # self.rel_tot = args.rel_tot
        self.in_path = in_path

        self.positive_triples = self.load_positive_triples()


        self.triple_index = self.create_triple_index()

        self.diffusion_model = diffusion_model

        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.trainer = GaussianDiffusionTrainer(diffusion_model, self.beta_1, self.beta_T, self.T )
        self.sampler = GaussianDiffusionSampler(diffusion_model, self.beta_1, self.beta_T, self.T )

    def load_positive_triples(self):
        positive_triples = set()
        train_file = os.path.join(self.in_path, "train2id.txt")

        if not os.path.exists(train_file):
            raise FileNotFoundError(f"训练文件不存在: {train_file}")

        with open(train_file, 'r') as f:
            lines = f.readlines()
            num_triples = int(lines[0].strip())
            for line in lines[1:1 + num_triples]:
                h, t, r = map(int, line.strip().split())
                positive_triples.add((h, r, t))

        return positive_triples

    def create_triple_index(self):

        triple_index = defaultdict(set)
        for h, r, t in self.positive_triples:
            triple_index[(h, r)].add(t)
            triple_index[(t, r)].add(h)
        return triple_index

    def is_positive_triple(self, h, r, t):

        return (h, r, t) in self.positive_triples


    def random_negative_sampling(self, batch_h, batch_r, batch_t):

        batch_size = batch_h.size(0)
        device = batch_h.device

        batch_neg_h = torch.randint(0, self.ent_tot, (batch_size,), device=device)
        batch_neg_t = torch.randint(0, self.ent_tot, (batch_size,), device=device)

        return batch_neg_h, batch_neg_t

    def normalize_embeddings(self, x):

        x_min = x.amin(dim=(2, 3), keepdim=True)  # [batch_size, channels, 1, 1]
        x_max = x.amax(dim=(2, 3), keepdim=True)  # [batch_size, channels, 1, 1]

        normalized_x = 2.0 * (x - x_min) / (x_max - x_min + 1e-8) - 1.0
        return normalized_x, x_min, x_max

    def denormalize_embeddings(self, normalized_x, x_min, x_max):

        original_x = x_min + (normalized_x + 1.0) * (x_max - x_min) / 2.0
        return original_x

    @staticmethod
    def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):


        assert v0.shape == v1.shape, "输入形状不一致"

        dot = torch.sum(v0 * v1 / (torch.norm(v0, dim=-1, keepdim=True) * torch.norm(v1, dim=-1, keepdim=True)))

        if torch.abs(dot) > DOT_THRESHOLD:
            return v0 + t * (v1 - v0)

        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)

        theta_t = theta_0 * t
        sin_theta_t = torch.sin(theta_t)

        s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0

        return s0 * v0 + s1 * v1

    @staticmethod
    def linear_interp(t, v0, v1):

        return (1 - t) * v0 + t * v1

    # def load_diffusion_model(self, args):
    #     from Diffusion.Model import UNet  # 假设模型定义在Diffusion.Model中
    #     from Diffusion import GaussianDiffusionSampler
    #
    #     self.diffusion_model = UNet(
    #         T=args.diffusion_config["T"],
    #         ch=args.diffusion_config["channel"],
    #         ch_mult=args.diffusion_config["channel_mult"],
    #         attn=args.diffusion_config["attn"],
    #         num_res_blocks=args.diffusion_config["num_res_blocks"],
    #         dropout=args.diffusion_config["dropout"]
    #     )
    #     state_dict = torch.load(args.diffusion_model_path, map_location='cpu')
    #     self.diffusion_model.load_state_dict(state_dict)
    #     self.diffusion_model.eval()
    #
    #     self.sampler = GaussianDiffusionSampler(
    #         self.diffusion_model,
    #         args.diffusion_config["beta_1"],
    #         args.diffusion_config["beta_T"],
    #         args.diffusion_config["T"]
    #     )
    #
    def interpolate_negative_samples(self, positive_embeddings, negative_embeddings,
                                     interpolation_method='linear', strength=0.5):

        if self.diffusion_model is None:
            raise RuntimeError("扩散模型未加载")

        batch_size, dim = positive_embeddings.shape
        device = positive_embeddings.device

        sqrt_dim = math.ceil(math.sqrt(dim))
        if sqrt_dim % 2 != 0:
            sqrt_dim += 1
        pad_dim = sqrt_dim * sqrt_dim - dim
        positive_embeddings = F.pad(positive_embeddings, (0, pad_dim), "constant", 0)
        negative_embeddings = F.pad(negative_embeddings, (0, pad_dim), "constant", 0)

        pos_2d = positive_embeddings.view(batch_size, 1, sqrt_dim, sqrt_dim)
        neg_2d = negative_embeddings.view(batch_size, 1, sqrt_dim, sqrt_dim)

        pos_2d, x_min_1, x_max_1 = self.normalize_embeddings(pos_2d)
        neg_2d, x_min_2, x_max_2 = self.normalize_embeddings(neg_2d)

        if interpolation_method == 'spherical':
            noise = self.slerp(strength,
                               torch.randn_like(pos_2d),
                               torch.randn_like(neg_2d))
        else:
            noise = self.linear_interp(strength,
                                       torch.randn_like(pos_2d),
                                       torch.randn_like(neg_2d))

        with torch.no_grad():
            sampled = self.sampler.sample_from_step(noise, from_step=self.T)

            sampled = self.denormalize_embeddings(sampled, torch.minimum(x_min_1, x_min_2), torch.maximum(x_max_1, x_max_2))

            sampled = sampled.view(batch_size, -1)[:, :dim]

        return sampled


    def train_diffusion(self, batch):
        if self.diffusion_model is None:
            raise RuntimeError("扩散模型未加载")

        batch_size, dim = batch.shape
        device = batch.device

        sqrt_dim = math.ceil(math.sqrt(dim))
        if sqrt_dim % 2 != 0:
            sqrt_dim += 1
        pad_dim = sqrt_dim * sqrt_dim - dim
        batch = F.pad(batch, (0, pad_dim), "constant", 0)

        pos_2d = batch.view(batch_size, 1, sqrt_dim, sqrt_dim)

        pos_2d, x_min, x_max = self.normalize_embeddings(pos_2d)

        loss = self.trainer(pos_2d).sum() / 1000

        return loss

