import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# ---------------- Positional Encoding ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)].to(x.device)

# ---------------- Transformer (sửa để match training) ----------------
class TransformerSeq2Seq(nn.Module):
    """
    Thiết kế sao cho forward(src_embedded, tgt_input_ids, src_attn_mask=None, tgt_attn_mask=None)
    - src_embedded: (B, S, E) — bạn có thể pass embedding matrix bên ngoài (embedding_src[src_ids])
    - tgt_input_ids: (B, T)  — token ids cho decoder input (BOS.. token_{n-1})
    - src_attn_mask / tgt_attn_mask: (B, S) / (B, T) with 1 for real tokens, 0 for pad
    """
    def __init__(self,
                 embed_dim,
                 vocab_size,                   # target vocab size (output dim)
                 embedding_decoder=None,       # pretrained weights (np array or torch.Tensor) or None
                 num_heads=2,
                 num_layers=2,
                 dim_feedforward=256,
                 dropout=0.1,
                 freeze_decoder_emb=True,
                 max_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)

        # decoder embedding (pretrained optional)
        if embedding_decoder is None:
            self.embedding_decoder = nn.Embedding(vocab_size, embed_dim)
        else:
            if not isinstance(embedding_decoder, torch.Tensor):
                embedding_decoder = torch.tensor(embedding_decoder, dtype=torch.float)
            self.embedding_decoder = nn.Embedding.from_pretrained(embedding_decoder, freeze=freeze_decoder_emb)

        # encoder/decoder (batch_first True -> inputs shape (B, T, E))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                                        batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads,
                                                        dim_feedforward=dim_feedforward, dropout=dropout,
                                                        batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, src_embedded, tgt_input_ids, src_attn_mask=None, tgt_attn_mask=None):
        """
        src_embedded : (B, S, E)
        tgt_input_ids: (B, T)
        src_attn_mask : (B, S) mask: 1 real token, 0 pad  (optional)
        tgt_attn_mask : (B, T) same
        """
        device = src_embedded.device
        # tgt embedding
        tgt_embedded = self.embedding_decoder(tgt_input_ids)  # (B, T, E)

        # add positional encoding
        src = self.pos_encoder(src_embedded)  # (B, S, E)
        tgt = self.pos_encoder(tgt_embedded)  # (B, T, E)

        # prepare key_padding_mask: True at positions that should be masked (pad positions)
        src_key_padding_mask = None
        tgt_key_padding_mask = None
        if src_attn_mask is not None:
            src_key_padding_mask = (src_attn_mask == 0).to(device)  # (B, S), bool
        if tgt_attn_mask is not None:
            tgt_key_padding_mask = (tgt_attn_mask == 0).to(device)  # (B, T)

        # encode
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)  # (B, S, E)

        # causal mask for decoder (T x T)
        T = tgt.size(1)
        if T > 0:
            tgt_mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        else:
            tgt_mask = None

        # decode
        output = self.decoder(tgt, memory,
                              tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=src_key_padding_mask)  # (B, T, E)

        logits = self.output_proj(output)  # (B, T, vocab)
        return logits

# ---------------- Helpers to apply embedding_src (tensor or nn.Embedding) ----------------
def apply_src_embedding(embedding_src, src_ids):
    """
    embedding_src can be:
      - torch.Tensor of shape (vocab_src, embed_dim)  -> indexing
      - nn.Embedding instance -> call( ids )
    src_ids: LongTensor (B, S)
    return: (B, S, E) float tensor on same device as src_ids
    """
    if isinstance(embedding_src, nn.Embedding):
        return embedding_src(src_ids)
    else:
        # assume it's a tensor/ndarray
        if not isinstance(embedding_src, torch.Tensor):
            embedding_src = torch.tensor(embedding_src, dtype=torch.float, device=src_ids.device)
        else:
            embedding_src = embedding_src.to(src_ids.device)
        return embedding_src[src_ids]
@torch.no_grad()
def translate(model, src_sentence, tokenizer_src, tokenizer_tgt, embedding_src, device, max_len=50):
    model.eval()
    inputs = tokenizer_src([src_sentence], return_tensors="pt", padding=True, truncation=True, max_length=128)
    src_ids = inputs["input_ids"].to(device)      # (1, S)
    src_attn = inputs.get("attention_mask", None)
    if src_attn is not None:
        src_attn = src_attn.to(device)

    src_embedded = apply_src_embedding(embedding_src, src_ids)  # (1, S, E)

    decoded_ids = [tokenizer_tgt.cls_token_id]
    for _ in range(max_len):
        decoder_input = torch.tensor([decoded_ids], device=device)
        # for decode we don't need tgt_attn_mask (we build causal mask inside model)
        logits = model(src_embedded, decoder_input, src_attn_mask=src_attn, tgt_attn_mask=None)
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        decoded_ids.append(next_token)
        if next_token == tokenizer_tgt.sep_token_id:
            break

    return tokenizer_tgt.decode(decoded_ids, skip_special_tokens=True)