import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .Model import Model


class AdvRelDistMult(Model):

    def __init__(
            self,
            ent_tot,
            rel_tot,
            dim=100,
            margin=6.0,
            epsilon=2.0,
            img_emb=None,
            text_emb=None,
            train2id_path=None
    ):

        super(AdvRelDistMult, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim
        self.dim_r = dim
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )
        self.img_dim = img_emb.shape[1]
        self.text_dim = text_emb.shape[1]
        self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
        self.text_embeddings = nn.Embedding.from_pretrained(text_emb).requires_grad_(False)

        # 1. 加载训练数据并计算关系基数性
        self.rel_categories = self.calculate_relation_categories(train2id_path, ent_tot, rel_tot)

        # 2. 关系嵌入投影层（将关系嵌入投影到与实体嵌入相同的维度）
        self.rel_proj = nn.Linear(self.dim_r, self.dim_e)

        # 多模态投影网络
        self.img_proj = nn.Sequential(
            nn.Linear(self.img_dim, self.dim_e),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(self.dim_e, self.dim_e)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.dim_e),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(self.dim_e, self.dim_e)
        )

        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )
        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )
        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data,
            a=-self.rel_embedding_range.item(),
            b=self.rel_embedding_range.item()
        )
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

        # 初始化投影层
        nn.init.xavier_uniform_(self.rel_proj.weight)
        nn.init.zeros_(self.rel_proj.bias)

        self.gate_linear = nn.Sequential(
            nn.Linear(self.dim_e * 2, self.dim_e),
            nn.Sigmoid()
        )

    # def get_joint_embeddings(self, es, ev, et, rg):
    #     e = torch.stack((es, ev, et), dim=1)
    #     u = torch.tanh(e)
    #     scores = self.ent_attn(u).squeeze(-1)
    #     attention_weights = torch.softmax(scores / torch.sigmoid(rg), dim=-1)  # Design of V8
    #     context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
    #     return context_vectors

    def calculate_relation_categories(self, train2id_path, ent_tot, rel_tot):
        """
        根据训练数据计算关系的基数性类别
        返回: 张量 [rel_tot] 包含每个关系的类别 (0:1-1, 1:1-n, 2:n-1, 3:n-n)
        """
        # 初始化数据结构
        head_count = [dict() for _ in range(rel_tot)]
        tail_count = [dict() for _ in range(rel_tot)]

        # 读取训练数据
        with open(train2id_path, 'r') as f:
            lines = f.readlines()
            # 跳过前两行（实体总数和关系总数）
            for line in lines[1:]:
                h, t, r = map(int, line.strip().split())
                # 统计每个(头实体, 关系)对应的尾实体数量
                if h not in head_count[r]:
                    head_count[r][h] = 0
                head_count[r][h] += 1

                # 统计每个(尾实体, 关系)对应的头实体数量
                if t not in tail_count[r]:
                    tail_count[r][t] = 0
                tail_count[r][t] += 1

        # 计算平均基数性
        rel_categories = torch.zeros(rel_tot, dtype=torch.long)
        for r in range(rel_tot):
            avg_head = np.mean(list(head_count[r].values())) if head_count[r] else 0
            avg_tail = np.mean(list(tail_count[r].values())) if tail_count[r] else 0

            # 分类关系
            if avg_head <= 1.5 and avg_tail <= 1.5:
                rel_categories[r] = 0  # 1-1
            elif avg_head <= 1.5 and avg_tail > 1.5:
                rel_categories[r] = 1  # 1-n
            elif avg_head > 1.5 and avg_tail <= 1.5:
                rel_categories[r] = 2  # n-1
            else:
                rel_categories[r] = 3  # n-n

        return rel_categories

    def get_joint_embeddings(self, es, ev, et, batch_r, is_head):
        batch_size = es.size(0)

        # 1. 获取关系类别
        rel_cat = self.rel_categories[batch_r.cpu()].to(es.device)

        # 2. 获取关系嵌入并投影
        rel_emb = self.rel_embeddings(batch_r)
        rel_emb_proj = self.rel_proj(rel_emb)

        # 3. 基础多模态融合
        joint_emb = (es + ev + et) / 3.0

        # 修复维度问题
        if rel_emb_proj.size(0) == 1 and batch_size > 1:
            rel_emb_proj = rel_emb_proj.expand(batch_size, -1)
            rel_cat = rel_cat.expand(batch_size)

        # 4. 根据关系类别和实体类型，使用门控机制动态融合关系信息
        mask = None
        if is_head:
            mask = (rel_cat == 2) | (rel_cat == 3)  # n-1或n-n
        else:
            mask = (rel_cat == 1) | (rel_cat == 3)  # 1-n或n-n

        # 只对需要融合的部分进行操作
        if mask is not None and mask.sum() > 0:
            # 提取需要融合的实体和关系嵌入
            masked_joint_emb = joint_emb[mask]
            masked_rel_emb = rel_emb_proj[mask]

            # --- 门控融合开始 ---
            gate_input = torch.cat([masked_joint_emb, masked_rel_emb], dim=-1)
            gate = self.gate_linear(gate_input)  # 计算门控值

            # 将门控值应用到关系嵌入上，然后加到实体嵌入上
            fused_info = gate * masked_rel_emb
            joint_emb[mask] = masked_joint_emb + fused_info
            # --- 门控融合结束 ---

        return joint_emb

    def cal_score(self, embs):
        return self._calc(embs[0], embs[2], embs[1], "")

    def _calc(self, h, t, r, mode):
        # DistMult 评分函数直接计算三向点积 h * r * t

        if mode == "head_batch":
            # 在 head_batch 模式下，h 的维度与其他不同，需要进行广播以匹配
            # h: [batch_size, 1, embedding_dim]
            # r: [1, num_relations, embedding_dim]
            # t: [1, num_relations, embedding_dim]
            #
            # 通过广播，计算 h * r * t
            # (h * r) 的结果维度为 [batch_size, num_relations, embedding_dim]
            # 然后再与 t 进行逐元素乘积
            score = h * r * t
        else:  # tail_batch and single mode
            # 在 tail_batch 或 single 模式下，h, r, t 的维度通常是一致的或可以广播的
            # h: [batch_size, embedding_dim]
            # r: [batch_size, embedding_dim]
            # t: [batch_size, embedding_dim]
            score = h * r * t

        # 沿嵌入维度（最后一个维度）求和，得到最终评分
        score = score.sum(dim=-1) * 10000
        return -score.flatten()

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        # rg = self.rel_gate(batch_r)
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, batch_r, is_head=True)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, batch_r, is_head=False)
        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score

    def forward_and_return_embs(self, data, epoch):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        # if epoch >= self.max_epoch / 2:
        #     batch_h, batch_t, batch_r = self.sampler.dynamic_negative_sampling(batch_h[:self.batch_size], batch_t[:self.batch_size], batch_r[:self.batch_size], self.neg_num)
        # if epoch >= self.max_epoch / 2:
        #     # 替换负样本（嵌入层作为参数传递）
        #     batch_h, batch_t, batch_r = self.sampler.replace_negatives(
        #         batch_h, batch_t, batch_r,
        #         self.img_embeddings, self.text_embeddings
        #     )

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        # rg = self.rel_gate(batch_r)
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, batch_r, is_head=True)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, batch_r, is_head=False)

        # # 训练扩散模型
        # self.train_diffusion(h_joint, r, t_joint)
        # # 生成负样本
        # batch_neg_h, batch_neg_t = self.generator.sample(h_joint[0:self.batch_size], r[0:self.batch_size], t_joint[0:self.batch_size])
        # # 替换上生成的负样本
        # h_joint, r, t_joint = self.replace_negative_samples(h_joint, r, t_joint, batch_neg_h, batch_neg_t)

        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score, [h_joint, r, t_joint]

    def replace_negative_samples(self, h_joint, r, t_joint, batch_neg_h, batch_neg_t):
        """
        Replace negative samples in the joint tensors with generated negative samples.

        Args:
            h_joint: tensor of shape (batch_size * (neg_num + 1), dim)
            r: tensor of shape (batch_size * (neg_num + 1), dim)
            t_joint: tensor of shape (batch_size * (neg_num + 1), dim)
            batch_neg_h: list of tensors, each of shape (batch_size, dim)
            batch_neg_t: list of tensors, each of shape (batch_size, dim)
        Returns:
            tuple: (h_joint_new, r, t_joint_new) with replaced negative samples
        """
        total_size = h_joint.size(0)
        batch_size = batch_neg_h[0].size(0)
        neg_num = total_size // batch_size - 1  # 计算每个正样本的负样本数
        G = len(batch_neg_h)  # 生成的负样本组数

        # 复制原始数据避免修改原tensor
        h_joint_new = h_joint.clone()
        t_joint_new = t_joint.clone()

        # 计算每个正样本负样本的起始索引
        base_indices = batch_size + torch.arange(batch_size, device=h_joint.device) * neg_num

        # 正样本的头实体和尾实体（用于构建负样本）
        h_pos = h_joint[:batch_size]  # (batch_size, dim)
        t_pos = t_joint[:batch_size]  # (batch_size, dim)

        # 遍历每组生成的负样本
        for g in range(G):
            # 获取当前组的生成负样本
            neg_h = batch_neg_h[g]  # (batch_size, dim)
            neg_t = batch_neg_t[g]  # (batch_size, dim)

            # 计算当前组的位置索引
            indices = base_indices + 2 * g

            # 替换头实体负样本（h负，r正，t正）
            h_joint_new[indices] = neg_h  # 替换头实体为生成的负样本头
            t_joint_new[indices] = t_pos  # 尾实体保持原正样本

            # 替换尾实体负样本（h正，r正，t负）
            h_joint_new[indices + 1] = h_pos  # 头实体保持原正样本
            t_joint_new[indices + 1] = neg_t  # 替换尾实体为生成的负样本尾

        return h_joint_new, r, t_joint_new

    def train_diffusion(self, batch_h, batch_r, batch_t):
        for epoch in range(self.g_epoch):
            self.optimizer_g.zero_grad()

            diff_loss = self.generator(batch_h, batch_r, batch_t)  # multimodal diffusion loss

            diff_loss.backward(retain_graph=True)
            self.optimizer_g.step()
            return diff_loss

    def get_batch_ent_embs(self, data):
        return self.ent_embeddings(data)

    def get_batch_vis_embs(self, data):
        return self.img_proj(self.img_embeddings(data))

    def get_batch_text_embs(self, data):
        return self.text_proj(self.text_embeddings(data))

    def get_batch_ent_multimodal_embs(self, data):
        return self.ent_embeddings(data), self.img_proj(self.img_embeddings(data)), self.text_proj(
            self.text_embeddings(data))

    def get_fake_score(
            self,
            batch_h,
            batch_r,
            batch_t,
            mode,
            fake_hv=None,
            fake_tv=None,
            fake_ht=None,
            fake_tt=None
    ):
        if fake_hv is None or fake_tv is None or fake_ht is None or fake_tt is None:
            raise NotImplementedError
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        # the fake joint embedding
        rg = self.rel_gate(batch_r)
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, rg)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, rg)
        h_fake = self.get_joint_embeddings(h, fake_hv, fake_ht, rg)
        t_fake = self.get_joint_embeddings(t, fake_tv, fake_tt, rg)
        score_h = self.margin - self._calc(h_fake, t_joint, r, mode)
        score_t = self.margin - self._calc(h_joint, t_fake, r, mode)
        score_all = self.margin - self._calc(h_fake, t_fake, r, mode)
        return [score_h, score_t, score_all], [h_fake, r, t_fake]

    def get_ns_score(self,
                     batch_h,
                     batch_r,
                     batch_t,
                     ns_h,
                     ns_t,
                     mode):
        score_h = self.margin - self._calc(ns_h, batch_t, batch_r, mode)
        score_t = self.margin - self._calc(batch_h, ns_t, batch_r, mode)
        return [score_h, score_t]

    def predict(self, data):
        score = -self.forward(data)
        return score.cpu().data.numpy()

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def get_attention(self, es, ev, et):
        # es, ev, et: [num_ent, emb_dim]
        e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=-1)
        return attention_weights

    def get_attention_weight(self, h, t):
        h = torch.LongTensor([h])
        t = torch.LongTensor([t])
        h_s = self.ent_embeddings(h)
        t_s = self.ent_embeddings(t)
        h_img_emb = self.img_proj(self.img_embeddings(h))
        t_img_emb = self.img_proj(self.img_embeddings(t))
        h_text_emb = self.text_proj(self.text_embeddings(h))
        t_text_emb = self.text_proj(self.text_embeddings(t))
        # the fake joint embedding
        h_attn = self.get_attention(h_s, h_img_emb, h_text_emb)
        t_attn = self.get_attention(t_s, t_img_emb, t_text_emb)
        return h_attn, t_attn

    def mm_negative_score(
            self,
            batch_h,
            batch_r,
            batch_t,
            mode,
            w_margin,
            neg_h=None,
            neg_t=None

    ):
        # h = self.ent_embeddings(batch_h)
        # t = self.ent_embeddings(batch_t)
        h = batch_h
        t = batch_t
        r = batch_r

        score_h = self._calc(neg_h, t, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
        score_t = self._calc(h, neg_t, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
        score_all = self._calc(neg_h, neg_t, r, mode).view(-1, batch_h.shape[0]).permute(1, 0)
        return [score_h, score_t, score_all]
