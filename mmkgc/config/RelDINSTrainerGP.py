# coding:utf-8
from calendar import c
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime

from tqdm import tqdm


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch
from sklearn.neighbors import NearestNeighbors


class RelDINSTrainerGP(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd",
                 save_steps=None,
                 checkpoint_dir=None,
                 generator=None,
                 lrg=None,
                 mu=None,
                 g_epoch=None,
                 sampler=None,
                 use_diffusion=False,
                 initial_ratio=0.8,
                 final_ratio=0.4
                 ):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha
        # learning rate of the generator
        assert lrg is not None
        self.alpha_g = lrg

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir


        assert mu is not None
        self.optimizer_g = None
        self.batch_size = self.model.batch_size
        self.mu = mu
        self.beta = 0.1

        self.g_epoch = g_epoch

        self.sampler = sampler
        self.use_diffusion = use_diffusion
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.warmup_epochs = train_times // 2

    def train_one_step(self, data, epoch):

        self.optimizer.zero_grad()
        loss, p_score, real_embs = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode'],
        }, epoch)


        batch_h_gen = self.to_var(data['batch_h'][0: self.batch_size], self.use_gpu)
        batch_t_gen = self.to_var(data['batch_t'][0: self.batch_size], self.use_gpu)
        batch_r = self.to_var(data['batch_r'][0: self.batch_size], self.use_gpu)
        batch_hs, batch_hv, batch_ht = self.model.model.get_batch_ent_multimodal_embs(batch_h_gen)
        batch_ts, batch_tv, batch_tt = self.model.model.get_batch_ent_multimodal_embs(batch_t_gen)
        batch_h_joint = self.model.model.get_joint_embeddings(batch_hs, batch_hv, batch_ht, batch_r,is_head=True)
        batch_t_joint = self.model.model.get_joint_embeddings(batch_ts, batch_tv, batch_tt, batch_r, is_head=False)

        loss_g = torch.tensor(0.0)
        if self.use_diffusion:
            self.optimizer_g.zero_grad()
            loss_h = self.sampler.train_diffusion(batch_h_joint)
            loss_t = self.sampler.train_diffusion(batch_t_joint)
            loss_g = loss_h + loss_t
            loss_g.backward(retain_graph=True)
            self.optimizer_g.step()

        batch_neg_h, batch_neg_t = self.sampler.random_negative_sampling(batch_h_gen, batch_r, batch_t_gen)
        batch_neg_hs, batch_neg_hv, batch_neg_ht = self.model.model.get_batch_ent_multimodal_embs(batch_neg_h)
        batch_neg_ts, batch_neg_tv, batch_neg_tt = self.model.model.get_batch_ent_multimodal_embs(batch_neg_t)
        batch_neg_h_joint = self.model.model.get_joint_embeddings(batch_neg_hs, batch_neg_hv, batch_neg_ht,  batch_r,is_head=True)
        batch_neg_t_joint = self.model.model.get_joint_embeddings(batch_neg_ts, batch_neg_tv, batch_neg_tt,  batch_r,is_head=False)
        interp_ratio = self.get_interpolation_ratio(epoch)

        if self.use_diffusion:
            batch_interp_h_joint = self.sampler.interpolate_negative_samples(batch_h_joint, batch_neg_h_joint, 'spherical', interp_ratio)
            batch_interp_t_joint = self.sampler.interpolate_negative_samples(batch_t_joint, batch_neg_t_joint, 'spherical', interp_ratio)
        else:
            batch_interp_h_joint = self.sampler.slerp(interp_ratio, batch_h_joint, batch_neg_h_joint)
            batch_interp_t_joint = self.sampler.slerp(interp_ratio, batch_t_joint, batch_neg_t_joint)

        # if epoch == 0:
        #     fig = self.visualize_focused_negative_sampling(
        #         batch_t_joint=batch_t_joint,
        #         batch_neg_t_joint=batch_neg_t_joint,
        #         batch_interp_t_joint=batch_interp_t_joint,
        #         n_samples=60,
        #         interp_strength=0.8,
        #         method='tsne',
        #         save_path='focused_negative_sampling_analysis_0.8.png'
        #     )

        scores = self.model.model.get_ns_score(
            batch_h=batch_h_joint,
            batch_r=real_embs[1][0: self.batch_size],
            batch_t=batch_t_joint,
            mode=data['mode'],
            ns_h=batch_interp_h_joint,
            ns_t=batch_interp_t_joint
        )
        for i, score in enumerate(scores):
            if i == 0:
                regul = (torch.mean(batch_interp_h_joint ** 2) +
                         torch.mean(real_embs[1][0: self.batch_size] ** 2) +
                         torch.mean(batch_t_joint ** 2)) / 3
            else:
                regul = (torch.mean(batch_h_joint ** 2) +
                         torch.mean(real_embs[1][0: self.batch_size] ** 2) +
                         torch.mean(batch_interp_t_joint ** 2)) / 3

            loss += self.model.loss(p_score, score) * self.mu
            loss += self.model.regul_rate * regul
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_g.item()


    def visualize_focused_negative_sampling(self, batch_t_joint, batch_neg_t_joint, batch_interp_t_joint,
                                            interp_strength=0.5, n_samples=50, method='tsne',
                                            save_path='focused_negative_sampling.png',
                                            figsize=(16, 11), dpi=300):

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.unicode_minus'] = False

        try:
            font_path = fm.findfont(fm.FontProperties(family='Times New Roman'))
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.serif'] = ['Times New Roman']
        except:
            print("Times New Roman not found, using default serif font")

        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                if x.is_cuda:
                    x = x.cpu()
                return x.detach().numpy()
            return x

        batch_t_joint = to_numpy(batch_t_joint)
        batch_neg_t_joint = to_numpy(batch_neg_t_joint)
        batch_interp_t_joint = to_numpy(batch_interp_t_joint)


        all_embeddings = np.vstack([batch_t_joint, batch_neg_t_joint, batch_interp_t_joint])

        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
            method_name = 't-SNE'
        else:
            reducer = PCA(n_components=2, random_state=42)
            method_name = 'PCA'

        all_embeddings_2d = reducer.fit_transform(all_embeddings)

        pos_embeddings_2d = all_embeddings_2d[:len(batch_t_joint)]

        nbrs = NearestNeighbors(n_neighbors=n_samples).fit(pos_embeddings_2d)
        distances, indices = nbrs.kneighbors(pos_embeddings_2d)

        avg_distances = np.mean(distances, axis=1)
        center_idx = np.argmin(avg_distances)

        closest_indices = indices[center_idx]

        selected_t = batch_t_joint[closest_indices]
        selected_neg = batch_neg_t_joint[closest_indices]
        selected_interp = batch_interp_t_joint[closest_indices]

        selected_embeddings = np.vstack([selected_t, selected_neg, selected_interp])

        if method == 'tsne':
            perplexity = min(30, n_samples - 1)
            focused_reducer = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, random_state=42)
        else:
            focused_reducer = PCA(n_components=2, random_state=42)

        focused_embeddings_2d = focused_reducer.fit_transform(selected_embeddings)

        labels = np.array([0] * n_samples + [1] * n_samples + [2] * n_samples)

        fig, ax = plt.subplots(figsize=figsize)

        colors = [ '#20B2AA','#FF7F50', '#1E90FF']
        markers = ['o', 'o', 'o']
        sizes = [250, 250, 250]

        labels_text = [
            'Positive Samples',
            'Random Negative Samples',
            f'Diffusion Interpolation Samples (ω={interp_strength:.2f})'
        ]

        start_idx = 0
        scatter_plots = []
        for i, (color, marker, label, size) in enumerate(zip(colors, markers, labels_text, sizes)):
            end_idx = start_idx + n_samples
            scatter = ax.scatter(
                focused_embeddings_2d[start_idx:end_idx, 0],
                focused_embeddings_2d[start_idx:end_idx, 1],
                c=color,
                marker=marker,
                alpha=0.8,
                s=size,
                edgecolors='none',
                linewidths=0,
                label=label
            )
            scatter_plots.append(scatter)
            start_idx = end_idx

        for i in range(min(n_samples, 50)):
            ax.plot([focused_embeddings_2d[i, 0], focused_embeddings_2d[i + n_samples * 2, 0]],
                    [focused_embeddings_2d[i, 1], focused_embeddings_2d[i + n_samples * 2, 1]],
                    color='#f7f99f',
                    linestyle='--',
                    alpha=0.6,
                    linewidth=2.0)

            ax.plot([focused_embeddings_2d[i, 0], focused_embeddings_2d[i + n_samples, 0]],
                    [focused_embeddings_2d[i, 1], focused_embeddings_2d[i + n_samples, 1]],
                    color='#f7f99f',
                    linestyle='--',
                    alpha=0.6,
                    linewidth=2.0)

        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
            labeltop=False,
            labelright=False
        )

        ax.grid(False)

        ax.set_frame_on(False)

        plt.tight_layout(pad=0)

        plt.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor='white')

        plt.show()

        return fig

    def calc_gradient_penalty(self, real_data, fake_data):
        batchsize = real_data[0].shape[0]
        alpha = torch.rand(batchsize, 1).cuda()
        inter_h = alpha * real_data[0].detach() + ((1 - alpha) * fake_data[0].detach())
        inter_r = alpha * real_data[1].detach() + ((1 - alpha) * fake_data[1].detach())
        inter_t = alpha * real_data[2].detach() + ((1 - alpha) * fake_data[2].detach())
        inter_h = torch.autograd.Variable(inter_h, requires_grad=True)
        inter_r = torch.autograd.Variable(inter_r, requires_grad=True)
        inter_t = torch.autograd.Variable(inter_t, requires_grad=True)
        inters = [inter_h, inter_r, inter_t]
        scores = self.model.model.cal_score(inters)

        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=inters,
            grad_outputs=torch.ones(scores.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.beta  # opt.GP_LAMBDA
        return gradient_penalty

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            self.optimizer_g = optim.Adam(
                self.sampler.diffusion_model.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )
            print(
                "Learning Rate of Model: {}, \nLearning Rate of Diffusion Model: {}".format(
                    self.alpha, self.alpha_g)
            )
        else:
            raise NotImplementedError
        print("Finish initializing...")

        # self.model.model.init_optimizer_g()

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:

            res = 0.0
            res_g = 0.0
            for data in self.data_loader:
                loss, loss_g = self.train_one_step(data, epoch)
                res += loss
                res_g += loss_g
            training_range.set_description("Epoch %d | Model loss: %f, Diffusion Model loss %f" % (epoch, res, res_g))

            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

    def get_interpolation_ratio(self, epoch):
        if epoch < self.warmup_epochs:
            return self.initial_ratio - (self.initial_ratio - self.final_ratio) * epoch / self.warmup_epochs
        return self.final_ratio

    def set_model(self, model):
        self.model = model



    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir


