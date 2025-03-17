I combined LSTM for local planning with Transformer for global planning


LT:
     input -> LSTM + TransformerEncoder -> out

LSTM:
     input ->  emb +  self-attn + LSTM + Linear -> out
     
TransformerEncoder:
    input -> emb 
