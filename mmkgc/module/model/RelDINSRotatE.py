import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .Model import Model
from .transformer_encoder import TransformerMultiModalEncoder


class RelDINSRotatE(Model):

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

        super(RelDINSRotatE, self).__init__(ent_tot, rel_tot)
        assert img_emb is not None
        assert text_emb is not None
        self.margin = margin
        self.epsilon = epsilon
        self.dim_e = dim * 2
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

        self.rel_categories = self.calculate_relation_categories(train2id_path, ent_tot, rel_tot)

        self.rel_proj = nn.Linear(self.dim_r, self.dim_e)

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

        nn.init.xavier_uniform_(self.rel_proj.weight)
        nn.init.zeros_(self.rel_proj.bias)

        self.gate_linear = nn.Sequential(
            nn.Linear(self.dim_e * 2, self.dim_e),
            nn.Sigmoid()
        )


    def calculate_relation_categories(self, train2id_path, ent_tot, rel_tot):

        head_count = [dict() for _ in range(rel_tot)]
        tail_count = [dict() for _ in range(rel_tot)]

        with open(train2id_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                h, t, r = map(int, line.strip().split())
                if h not in head_count[r]:
                    head_count[r][h] = 0
                head_count[r][h] += 1

                if t not in tail_count[r]:
                    tail_count[r][t] = 0
                tail_count[r][t] += 1

        rel_categories = torch.zeros(rel_tot, dtype=torch.long)
        for r in range(rel_tot):
            avg_head = np.mean(list(head_count[r].values())) if head_count[r] else 0
            avg_tail = np.mean(list(tail_count[r].values())) if tail_count[r] else 0

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

        rel_cat = self.rel_categories[batch_r.cpu()].to(es.device)

        rel_emb = self.rel_embeddings(batch_r)
        rel_emb_proj = self.rel_proj(rel_emb)

        joint_emb = (es + ev + et) / 3.0

        if rel_emb_proj.size(0) == 1 and batch_size > 1:
            rel_emb_proj = rel_emb_proj.expand(batch_size, -1)
            rel_cat = rel_cat.expand(batch_size)

        mask = None
        if is_head:
            mask = (rel_cat == 2) | (rel_cat == 3)
        else:
            mask = (rel_cat == 1) | (rel_cat == 3)

        if mask is not None and mask.sum() > 0:
            masked_joint_emb = joint_emb[mask]
            masked_rel_emb = rel_emb_proj[mask]

            gate_input = torch.cat([masked_joint_emb, masked_rel_emb], dim=-1)
            gate = self.gate_linear(gate_input)  # 计算门控值

            fused_info = gate * masked_rel_emb
            joint_emb[mask] = masked_joint_emb + fused_info

        return joint_emb

    def cal_score(self, embs):
        return self._calc(embs[0], embs[2], embs[1], "")

    def _calc(self, h, t, r, mode):
        pi = self.pi_const

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1,
                               re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1,
                               re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1,
                               re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1,
                               re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(
            -1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(
            -1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)
        return score.permute(1, 0).flatten()

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
        return score, [h_joint, r, t_joint]

    def replace_negative_samples(self, h_joint, r, t_joint, batch_neg_h, batch_neg_t):
        total_size = h_joint.size(0)
        batch_size = batch_neg_h[0].size(0)
        neg_num = total_size // batch_size - 1
        G = len(batch_neg_h)

        h_joint_new = h_joint.clone()
        t_joint_new = t_joint.clone()

        base_indices = batch_size + torch.arange(batch_size, device=h_joint.device) * neg_num

        h_pos = h_joint[:batch_size]  # (batch_size, dim)
        t_pos = t_joint[:batch_size]  # (batch_size, dim)

        for g in range(G):
            neg_h = batch_neg_h[g]  # (batch_size, dim)
            neg_t = batch_neg_t[g]  # (batch_size, dim)

            indices = base_indices + 2 * g

            h_joint_new[indices] = neg_h
            t_joint_new[indices] = t_pos

            h_joint_new[indices + 1] = h_pos
            t_joint_new[indices + 1] = neg_t

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
