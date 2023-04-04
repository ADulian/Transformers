import torch
import torch as nn

# --------------------------------------------------------------------------------
class Transformer(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, embed_size, num_heads, forward_expansion):
        super().__init__()

        self.attention = Attention(embed_size=embed_size, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Mapping to higher->original space size
        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion * embed_size),
                                          nn.ReLU(),
                                          nn.Linear(forward * embed_size, embed_size))

    # --------------------------------------------------------------------------------
    def forward(self, values, keys, query, mask):
        # Multi-Head Attention
        attention = self.attention(values, keys, query, mask)

        # Add & Norm
        residuals = attention + query
        residuals = self.norm1(skip_connection)

        # Feed Forward
        out = self.feed_forward(out)

        # Add & Norm
        residuals = residuals + out
        out = self.norm2(out)

        return out

# --------------------------------------------------------------------------------
class Attention(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, embed_size, num_heads):
        super().__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert (self.head_dim * num_heads == embedSize), "Embedding size needs to be divisible by num of heads"

        # --- Layers
        self.linear_value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear_query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.linear_keys = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(num_heads*self.head_dim, embed_size)


    # --------------------------------------------------------------------------------
    def forward(self, values, keys, query, mask):
        # Batch Size
        N = query.shape[0]

        #
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into num_heads pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = query.reshape(N, query_len_len, self.num_heads, self.head_dim)

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
        out = out.reshape(N, query_len, self.heads*self.head_dim)

        # Run through Lienar layer
        out = self.fc_out(out)

        return out

# --------------------------------------------------------------------------------
class Encoder(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self, vocab_size, embed_size, num_layers, num_heads,
                 device, forward_expansion, max_length):
        super().__init__()

        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                Transformer(embed_size=embed_size, num_heads=num_heads, forward_expansion=forward_expansion)
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

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# --------------------------------------------------------------------------------
class Decoder(nn.Module):

    # --------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

