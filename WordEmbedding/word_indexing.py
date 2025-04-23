import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd

if __name__ == '__main__':
    word_to_idx =  {"hello": 0 , "world": 1}
    embeds = nn.Embedding(2, 5) # 2 words in vocab, 5 dimensional embeddings
    print(embeds)
    lookup_tensor = torch.tensor([word_to_idx["world"]], dtype=torch.long)
    hello_embed = embeds(lookup_tensor)
    print(hello_embed)