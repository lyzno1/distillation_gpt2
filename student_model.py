import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

@dataclass
class StudentConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 6      # 减少层数
    n_head: int = 8       # 减少注意力头
    n_embd: int = 384     # 减少嵌入维度

class StudentTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 输入嵌入
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(0.1)
        
        # transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        # 获取位置编码
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # token和位置嵌入
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        
        # 组合并添加dropout
        x = self.drop(tok_emb + pos_emb)
        
        # transformer块
        for block in self.blocks:
            x = block(x)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        # 如果提供了目标，计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, T, C = x.size()
        
        # 计算注意力
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # 高效注意力计算
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # 重组输出
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x 