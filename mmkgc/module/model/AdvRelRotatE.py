import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import optim

from .Model import Model


class AdvRelRotatE(Model):

    def __init__(
        self,
        ent_tot,
        rel_tot,
        dim=100,
        margin=6.0,
        epsilon=2.0,
        img_emb=None,
        text_emb=None,
        generator=None,
        sampler=None
    ):

        super(AdvRelRotatE, self).__init__(ent_tot, rel_tot)
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
        # self.img_proj = nn.Linear(self.img_dim, self.dim_e)
        # self.text_proj = nn.Linear(self.text_dim, self.dim_e)
        self.img_proj = nn.Sequential(
            nn.Linear(self.img_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.dim_e),
            nn.ReLU(),
            nn.Linear(self.dim_e, self.dim_e)
        )

        self.ent_attn = nn.Linear(self.dim_e, 1, bias=False)
        self.ent_attn.requires_grad_(True)
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

        self.rel_gate = nn.Embedding(self.rel_tot, 1)
        nn.init.uniform_(
            tensor=self.rel_gate.weight.data,
            a=-self.ent_embedding_range.item(),
            b=self.ent_embedding_range.item()
        )

        # self.adv_scores = nn.Sequential(
        #     nn.Linear(self.dim_e, self.dim_e),
        #     nn.ReLU(),
        #     nn.Linear(self.dim_e, 1)
        # )


        self.g_epoch = 100
        self.batch_size = 512
        self.neg_num = 128
        self.alpha_g = 1e-4
        self.generator = generator
        self.optimizer_g = None
        self.generator.cuda()
        self.weight_decay = 0.0

        self.sampler = sampler

        self.max_epoch = 100



    def init_optimizer_g(self):
        self.optimizer_g = optim.Adam(
                self.generator.parameters(),
                lr=self.alpha_g,
                weight_decay=self.weight_decay,
            )


    def get_joint_embeddings(self, es, ev, et, rg):
        e = torch.stack((es, ev, et), dim=1)
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)
        attention_weights = torch.softmax(scores / torch.sigmoid(rg), dim=-1) # Design of V8
        context_vectors = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
        return context_vectors

    
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
        rg = self.rel_gate(batch_r)
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, rg)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, rg)
        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score
    
    def forward_and_return_embs(self, data, epoch):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']
        # if epoch >= self.max_epoch / 2:
        #     batch_h, batch_t, batch_r = self.sampler.dynamic_negative_sampling(batch_h[:self.batch_size], batch_t[:self.batch_size], batch_r[:self.batch_size], self.neg_num)
        if epoch >= self.max_epoch / 2:
            # 替换负样本（嵌入层作为参数传递）
            batch_h, batch_t, batch_r = self.sampler.replace_negatives(
                batch_h, batch_t, batch_r,
                self.img_embeddings, self.text_embeddings
            )

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(batch_h))
        t_img_emb = self.img_proj(self.img_embeddings(batch_t))
        h_text_emb = self.text_proj(self.text_embeddings(batch_h))
        t_text_emb = self.text_proj(self.text_embeddings(batch_t))
        rg = self.rel_gate(batch_r)
        h_joint = self.get_joint_embeddings(h, h_img_emb, h_text_emb, rg)
        t_joint = self.get_joint_embeddings(t, t_img_emb, t_text_emb, rg)

        # # 训练扩散模型
        # self.train_diffusion(h_joint, r, t_joint)
        # # 生成负样本
        # batch_neg_h, batch_neg_t = self.generator.sample(h_joint[0:self.batch_size], r[0:self.batch_size], t_joint[0:self.batch_size])
        # # 替换上生成的负样本
        # h_joint, r, t_joint = self.replace_negative_samples(h_joint, r, t_joint, batch_neg_h, batch_neg_t)

        score = self.margin - self._calc(h_joint, t_joint, r, mode)
        return score, [h_joint, r, t_joint]


    def replace_negative_samples(self,h_joint, r, t_joint, batch_neg_h, batch_neg_t):

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
        return self.ent_embeddings(data), self.img_proj(self.img_embeddings(data)), self.text_proj(self.text_embeddings(data))
    
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
