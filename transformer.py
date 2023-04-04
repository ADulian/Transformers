import torch
import torch as nn

# --------------------------------------------------------------------------------
class Transformer(nn.Module):
    # --------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

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



