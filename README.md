I wanna play rythm game osu but i cannot afford the device so i write this decrease Zero Drift
This is a Lightweight pytorch-based model which is created by myself and used for forecasting the time series

I combined LSTM with Transformer for both local and global planning

The structure is like this 

input : [bs, seq]

LSTM:
     input ->  emb + self-attn + LSTM + Linear -> out [bs,seq,class]
     
TransformerEncoder:
    input -> emb + PositionalEncoding + TransformerEncoder + Linear -> out [bs,seq,class]

LT:
     input -> cat( LSTM + TransformerEncoder, dim = -1 ) -> Linear to seq -> out [bs * seq , class]

activation use mish
so u get the data and u can convert it into int or index in char_list

What I wanna do is using LSTM for short area prediction and Transformer for the global prediction
Because the Transformer's MLA is better for the long range relationship prediction and the LSTM could fix the Global error

Such as earth spinning , Transformer was the sun controller and LSTM was earth controller.
Long range means the inputs may happend long before, area means the inputs happed now.
