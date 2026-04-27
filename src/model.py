from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.config import ProjectConfig


class Head(nn.Module):
    """Одна голова masked self-attention."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, time_steps, _ = x.shape

        # 1) Линейные проекции во внутреннее пространство головы: Q, K, V.
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # 2) Считаем scores = Q @ K^T.
        scores = q @ k.transpose(-2, -1)  # (B, T, T)

        # 3) Масштабируем на sqrt(head_size) для стабильности.
        scores = scores / math.sqrt(k.size(-1))

        # 4) Применяем causal mask: будущие токены недоступны.
        mask = self.tril[:time_steps, :time_steps]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5) Превращаем scores в вероятности внимания.
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # 6) Получаем выход головы как взвешенную сумму V.
        out = weights @ v  # (B, T, head_size)

        if return_attention:
            return out, weights
        return out, None


class MultiHeadAttention(nn.Module):
    """Параллельный набор attention-голов."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd должен делиться на n_head без остатка.")

        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd=n_embd, head_size=head_size, block_size=block_size, dropout=dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        head_outputs: List[torch.Tensor] = []
        head_attentions: List[torch.Tensor] = []

        for head in self.heads:
            out, attention = head(x, return_attention=return_attention)
            head_outputs.append(out)
            if return_attention and attention is not None:
                head_attentions.append(attention)

        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        if return_attention:
            # (B, heads, T, T)
            attentions = torch.stack(head_attentions, dim=1)
            return out, attentions
        return out, None


class FeedForward(nn.Module):
    """Двухслойный MLP внутри Transformer-блока."""

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Один Transformer-блок: attention + feed-forward."""

    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.sa = MultiHeadAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
        )
        self.ffwd = FeedForward(config.n_embd, config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        attention_out, attention_map = self.sa(self.ln1(x), return_attention=return_attention)
        x = x + attention_out  # residual connection
        x = x + self.ffwd(self.ln2(x))  # residual connection
        if return_attention:
            return x, attention_map
        return x, None


class TinyTransformer(nn.Module):
    """Маленькая модель для посимвольного моделирования текста."""

    def __init__(self, config: ProjectConfig):
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size должен быть > 0 перед созданием модели.")

        self.config = config
        self.block_size = config.block_size

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, List[torch.Tensor] | None]:
        batch_size, time_steps = idx.shape
        if time_steps > self.block_size:
            raise ValueError(f"Длина последовательности {time_steps} больше block_size={self.block_size}.")

        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        positions = torch.arange(time_steps, device=idx.device)
        position_embeddings = self.position_embedding_table(positions)[None, :, :]  # (1, T, C)
        x = token_embeddings + position_embeddings

        attention_maps: List[torch.Tensor] = []
        for block in self.blocks:
            x, block_attention = block(x, return_attention=return_attentions)
            if return_attentions and block_attention is not None:
                attention_maps.append(block_attention)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            logits_flat = logits.view(batch_size * time_steps, -1)
            targets_flat = targets.view(batch_size * time_steps)
            loss = F.cross_entropy(logits_flat, targets_flat)

        if return_attentions:
            return logits, loss, attention_maps
        return logits, loss, None

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8) -> torch.Tensor:
        temperature = max(temperature, 1e-5)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

