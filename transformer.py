import torch
import torch.nn as nn

# --------------------------------------------------------------------------------
class Transformer(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx,
                 embed_size=256, num_layers=6, forward_expansion=4, num_heads=8,
                 device="cuda", max_length=100):
        super().__init__()

        self.encoder = Encoder(src_vocab_size=src_vocab_size, embed_size=embed_size,
                               num_layers=num_layers, num_heads=num_heads,
                               device=device, forward_expansion=forward_expansion, max_length=max_length)

        self.decoder = Decoder(trg_vocab_size=trg_vocab_size, embed_size=embed_size,
                               num_layers=num_layers, num_heads=num_heads,
                               device=device, forward_expansion=forward_expansion, max_length=max_length)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.device = device

    # --------------------------------------------------------------------------------
    def forward(self, src, trg):
        mask_src = self.get_src_mask(src)
        mask_trg = self.get_trg_mask(trg)

        enc_src = self.encoder(src, mask_src)
        out = self.decoder(trg, enc_src, mask_src, mask_trg)

        return out

    # --------------------------------------------------------------------------------
    def get_src_mask(self, src):
        # (N, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask.to(self.device)

    # --------------------------------------------------------------------------------
    def get_trg_mask(self, trg):
        N, trg_len = trg.shape

        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)

        return trg_mask.to(self.device)

# --------------------------------------------------------------------------------
class Transformer_Block(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, embed_size, num_heads, forward_expansion):
        super().__init__()

        self.attention = Attention_Block(embed_size=embed_size, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Mapping to higher->original space size
        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion * embed_size),
                                          nn.ReLU(),
                                          nn.Linear(forward_expansion * embed_size, embed_size))

    # --------------------------------------------------------------------------------
    def forward(self, values, keys, query, mask):
        # Multi-Head Attention
        attention = self.attention(values, keys, query, mask)

        # Add & Norm
        residuals = attention + query
        residuals = self.norm1(residuals)

        # Feed Forward
        out = self.feed_forward(residuals)

        # Add & Norm
        residuals = residuals + out
        out = self.norm2(residuals)

        return out

# --------------------------------------------------------------------------------
class Attention_Block(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, embed_size, num_heads):
        super().__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (self.head_dim * num_heads == embed_size), "Embedding size needs to be divisible by num of heads"

        # --- Layers
        self.linear_value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear_query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear_keys = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(num_heads*self.head_dim, embed_size)


    # --------------------------------------------------------------------------------
    def forward(self, values, keys, query, mask):
        # Batch Size
        N = query.shape[0]

        # Shapes
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into num_heads pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len, self.num_heads, self.head_dim)

        # Send through linear layers
        values = self.linear_value(values)
        kets = self.linear_keys(keys)
        queries = self.linear_query(queries)

        # Query * Keys
        similarity = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Apply mask
        if mask is not None:
            similarity = similarity.masked_fill(mask == 0, float("-inf"))

        # Attention Filter
        attention = torch.softmax(similarity / (self.embed_size ** (1/2)), dim=3)

        # Attention * Values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        # Stack heads
        out = out.reshape(N, query_len, self.num_heads*self.head_dim)

        # Run through Lienar layer
        out = self.fc_out(out)

        return out

# --------------------------------------------------------------------------------
class Decoder_Block(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, embed_size, num_heads, forward_expansion, device):
        super().__init__()

        self.attention = Attention_Block(embed_size=embed_size, num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_size)

        self.transformer = Transformer_Block(embed_size=embed_size, num_heads=num_heads, forward_expansion=forward_expansion)

    # --------------------------------------------------------------------------------
    def forward(self, x, value, key, src_mask, trg_mask):
        # Masked Multi-Head Attention
        attention = self.attention(x, x, x, trg_mask)

        # Add & Norm
        residuals = attention + x
        query = self.norm(residuals)

        # 2nd Part of the decoder, the Transformer
        out = self.transformer(value, key, query, src_mask)

        return out

# --------------------------------------------------------------------------------
class Encoder(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, src_vocab_size, embed_size, num_layers, num_heads,
                 device, forward_expansion, max_length):
        super().__init__()

        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                Transformer_Block(embed_size=embed_size, num_heads=num_heads, forward_expansion=forward_expansion)
                for _ in range(num_layers)
            ]
        )

    # --------------------------------------------------------------------------------
    def forward(self, x, mask):
        N, seq_length = x.shape

        # Input Embedding
        emb_in = self.word_embedding(x)

        # Positional Embedding
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        emb_pos = self.position_embedding(positions)

        # Positional Encoding
        out = emb_in + emb_pos

        # Transformer
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# --------------------------------------------------------------------------------
class Decoder(nn.Module):

    # ----------------------------------------------------------------------------
    def __init__(self, trg_vocab_size, embed_size, num_layers, num_heads,
                 forward_expansion, device, max_length):
        super().__init__()

        self.device = device

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                Decoder_Block(embed_size=embed_size, num_heads=num_heads,
                              forward_expansion=forward_expansion, device=device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

    # --------------------------------------------------------------------------------
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape

        # Input Embedding
        emb_in = self.word_embedding(x)

        # Positional Embedding
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        emb_pos = self.position_embedding(positions)

        # Positional Encoding
        out = emb_in + emb_pos

        # Transformer
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)

        # FC
        out =self.fc_out(out)

        return out



