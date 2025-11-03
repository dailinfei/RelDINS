import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerMultiModalEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerMultiModalEncoder, self).__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

    def forward(self, es, ev, et):
        # 假设es, ev, et的形状都是 [batch_size, dim]
        # 拼接成 [seq_len, batch_size, dim]，其中 seq_len = 3
        x = torch.stack([es, ev, et], dim=0) # [3, batch_size, dim]

        # 添加位置编码
        x = self.positional_encoding(x)

        # 经过Transformer Encoder
        output = self.transformer_encoder(x) # [3, batch_size, dim]

        # 重新分离出es', ev', et'
        es_prime = output[0]
        ev_prime = output[1]
        et_prime = output[2]

        return es_prime, ev_prime, et_prime