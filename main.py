import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class TextEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, num_layers_gpt=4, num_heads_gpt=32, n_ctx=2048, n_embd=768, num_layers_decoder=2, num_heads_decoder=32):
        super(TextEmbeddingModel, self).__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size,  # размер словаря
            n_positions=n_ctx,   # максимальное количество позиций
            n_ctx=n_ctx,         # контекст
            n_embd=n_embd,         # размер эмбеддинга
            n_layer=num_layers_gpt,         # количество слоев
            n_head=num_heads_gpt
        )
        self.gpt = GPT2Model(config=self.config)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(n_embd, num_heads_decoder), num_layers_decoder)
        self.initial_embedding = nn.Parameter(torch.zeros(1, self.config.n_embd), requires_grad=True)

    def forward(self, x):
        x = self.gpt(x).last_hidden_state
        x = x[0]
        x = self.decoder(self.initial_embedding.data, x)
        return x
