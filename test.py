import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import LT

idx_char = ["e","h","l","o"]
x= [3,1,2,3,2]
y= [1,0,2,2,3]
batch_size = 1#构造h0的时候才需要
num_class = 4
seq_len = len(x)
input_size = len(idx_char)  # 输入词汇表大小
output_size = len(idx_char) # 输出词汇表大小

num_layers = 2#transform的层数
num_classes = 4
nhead = 2
hidden_size = 4 * nhead
seq_len_x = len(x)
seq_len_y = len(y)
embedding_size = nhead * 8 #数据嵌入层
inputs = torch.LongTensor(x).view(batch_size, seq_len)
ans = torch.LongTensor(y[:-1]).unsqueeze(0)
labels = torch.LongTensor(y).unsqueeze(0)


model = LT(input_size, output_size, hidden_size,
           num_layers, nhead,
           seq_len_x, seq_len_y,
           embedding_size,num_classes,batch_size)
device = torch.device("cuda")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)

def tarin(inputs,ans,lables):
    inputs,labels,ans = inputs.to(device),lables.to(device),ans.to(device)
    for epoch in range(200):
        optimizer.zero_grad()#记得清零
        #开始训练网络
        locals = model(inputs,ans)
        loss = criterion(locals.view(-1, num_class), labels.view(-1)) #batch_size * seq_len,num_calss
        loss.backward()#所有的求和后反向传递
        optimizer.step()#参数更新
        if epoch % 10 == 0:
            _,idx = locals.max(-1)
            idx = idx.cpu()
            idx = idx.data.numpy()
            print("".join([idx_char[x] for x in idx[0]]), end ="")
            print(' ,Epoch [%d], Loss: %.6f' % (epoch, loss.item()))

tarin(inputs,ans,labels)
