This is a simple pytorch-based model which is created by myself and used for forecasting the time series

I combined LSTM with Transformer for both local and global planning

The structure is like this 

LSTM:
     input ->  emb +  self-attn + LSTM + Linear -> out
     
TransformerEncoder:
    input -> emb + PositionalEncoding +  TransformerEncoder + Linear -> out

LT:
     input -> cat(LSTM + TransformerEncoder) -> Linear to seq -> out

What I want do is use LSTM for short area prediction and use Transformer for the global prediction
Because the Transformer's MLA is better for the long time relationship prediction
and the LSTM could fix the 

