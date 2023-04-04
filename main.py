"""Transformers from scratch
https://www.youtube.com/watch?v=U0s0f995w14&ab_channel=AladdinPersson
"""

import torch
from transformer import Transformer

# --------------------------------------------------------------------------------
def main():
    device = torch.device("cuda")

    x = torch.tensor([[1,5,6,4,3,9,5,2,0],
                      [1,8,7,3,4,5,6,7,2]], device=device)

    trg = torch.tensor([[1,7,4,3,5,9,2,0],
                        [1,5,6,2,4,7,6,2]], device=device)

    src_pad_idx = 0
    trg_pad_idx = 0

    src_vocab_size = 10
    trg_vocab_size = 10

    model = Transformer(src_vocab_size=src_vocab_size, trg_vocab_size=trg_vocab_size,
                        src_pad_idx=src_pad_idx, trg_pad_idx=trg_pad_idx).to(device)

    out = model(x, trg[:, :-1])

    print(out.shape)

# --------------------------------------------------------------------------------
if __name__ == '__main__':

    main()
