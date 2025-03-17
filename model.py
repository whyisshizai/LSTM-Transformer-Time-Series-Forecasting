import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from open_clip.transformer import LayerNorm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
    #使用动态的方法，可以处理任意长度的序列
    def forward(self, x):
        max_len = x.size(1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #x: [batch_size, seq_len, embedding_size]
        #pe:[batch,seq,emb]
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))

        pe = torch.zeros(1, max_len, self.d_model, device=x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return x + pe

class simple_Attention(nn.Module):
    def __init__(self, emb_size):
        super().__init__()

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size,emb_size)

    def forward(self, q, k, v):
        Q = self.query(q)
        K = self.key(k)
        V = self.value(v)

        #点积注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        #乘积
        output = torch.matmul(attn_weights, V)
        return output
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers,embbending_size,num_class):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embbending_size = embbending_size
        self.emb = nn.Embedding(input_size, embbending_size)
        self.input_size = input_size
        self.num_class = num_class
        self.num_layers = num_layers
        self.attention = simple_Attention(emb_size=embbending_size)
        self.lstm= nn.LSTM(input_size = embbending_size,
                           hidden_size = hidden_size,
                           num_layers = num_layers,
                           batch_first=True)
        self.hidden = (
            torch.zeros(self.num_layers,
                        self.batch_size,
                        self.hidden_size),
            torch.zeros(self.num_layers,
                        self.batch_size,
                        self.hidden_size),
        )
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, input):
        input = self.emb(input)
        input = self.attention(input, input,input)
        hidden = (self.hidden[0].to(next(self.parameters()).device),self.hidden[1].to(next(self.parameters()).device)) if next(self.parameters()).device != self.hidden[0].device else self.hidden
        out, _ = self.lstm(input,hidden)
        out = self.fc(out)
        return out

class LT(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, nhead, seq_len_x, seq_len_y, embedding_size,num_class,
                 batch_size = None,activation = None):
        super(LT, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_class = num_class
        self.activate = activation if activation else F.mish
        self.nhead = nhead

        self.MLA = nn.MultiheadAttention(embedding_size,num_heads=nhead)

        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        self.embedding_size = embedding_size

        # Embedding 层
        self.enc_emb = nn.Embedding(input_size, embedding_size)
        self.dec_emb = nn.Embedding(output_size, embedding_size)

        self.LSTM = LSTM(input_size = input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_size=batch_size,
                         embbending_size=embedding_size,
                         num_class = num_class
                         )

        #在这里已经有MLA了
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_size, nhead),
            num_layers,
            norm=LayerNorm(embedding_size),
        )


        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embedding_size, nhead),
            num_layers,
            norm=LayerNorm(embedding_size)
        )
        self.f2 = nn.Linear(embedding_size, num_class)


        self.enc_pos_encoder = PositionalEncoding(embedding_size)
        #文本分类任务就不需要
        self.dec_pos_encoder = PositionalEncoding(embedding_size)

    def forward(self,raw,cook=None,y=None):


        #本地规划
        raw_local = self.LSTM(raw)
        raw_local = self.activate(raw_local)

        raw = self.enc_emb(raw) * (self.embedding_size ** 0.5)
        raw = self.enc_pos_encoder(raw)


        raw_code = self.transformer_encoder(raw)

        # #全局规划
        # # cook = self.dec_emb(cook) * (self.embedding_size ** 0.5)
        # # cook = self.dec_pos_encoder(cook)
        # # raw_global = self.transformer_decoder(cook,raw_code)
        #
        raw_global = self.activate(self.f2(raw_code))
        #
        # return raw_local,raw_global
        out = torch.cat((raw_local,raw_global),dim=-1)
        return raw_local

