import torch
from torch import Tensor

from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.embedding import Embedding
from cs336_basics.ffnn import FFN
from cs336_basics.linear import Linear
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.softmax import softmax


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int | None = None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len)
        self.attn_norm = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
        self.ffn_norm = RMSNorm(d_model)
        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x: Tensor):
        attn_layer = x + self.attn.forward(self.attn_norm.forward(x))
        ffn_layer = attn_layer + self.ffn.forward(self.ffn_norm(attn_layer))
        return ffn_layer


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformers: list[TransformerBlock] = []
        for _ in range(num_layers):
            self.transformers.append(TransformerBlock(d_model, num_heads, d_ff, context_length))
        self.norm = RMSNorm(d_model)
        self.project = Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        embedding_out = self.embedding.forward(x)
        transformer_out = embedding_out
        for t in self.transformers:
            transformer_out = t.forward(transformer_out)
        norm_out = self.norm.forward(transformer_out)
        l_out = self.project.forward(norm_out)
        return l_out

    def decode(self, x: Tensor, max_toks: int, temp: float = 1.0) -> Tensor:
        with torch.no_grad():
            count = 0
            while count < max_toks:
                pred = self.forward(x)
                last_vocab_probs = softmax(pred[..., -1, :], -1, temp)

                dist = torch.distributions.Categorical(probs=last_vocab_probs)
                tok_id = dist.sample()  # need to reshape? looks like it is returning a flat list

                if tok_id == end_of_text_tok:  # how to get this?
                    break
                # TODO: nucleus top p sampling

                x = torch.stack([x, tok_id], dim=-2)
                count += 1

            return x
