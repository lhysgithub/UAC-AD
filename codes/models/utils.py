import torch
from torch import nn
from torch.autograd import Variable
import math
import numpy as np

def adjustment_decision(pred,gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred

def adjust_learning_rate(optimizer, lr_):
    lr = 0.8*lr_
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    return lr
        

def get_anomaly_scopes(y_test):
    index_list = []
    lens_list = []
    find = 0
    count = 0
    for i in range(len(y_test)):
        if int(y_test[i]) == 1:
            if find == 0:
                index_list.append(i)
            find = 1
            count += 1
        elif find == 1:
            find = 0
            lens_list.append(count)
            count = 0
    return index_list, lens_list

class Decoder(nn.Module):
    def __init__(self, encoded_dim, T, **kwargs):
        super(Decoder, self).__init__()
        linear_size = kwargs["linear_size"]

        layers = []
        for i in range(kwargs["decoder_layer_num"]-1):
            input_size = encoded_dim if i == 0 else linear_size
            layers += [nn.Linear(input_size, linear_size), nn.ReLU()]
        layers += [nn.Linear(linear_size, 2)]
        self.net = nn.Sequential(*layers)
        
        self.self_attention = kwargs["self_attention"]
        if self.self_attention:
            self.attn = SelfAttention(encoded_dim, T)

    def forward(self, x: torch.Tensor): #[batch_size, T, hidden_size*dir_num]
        if self.self_attention: ret = self.attn(x)
        else: ret = x[:, -1, :]
        return self.net(ret)

class Embedder(nn.Module):
    def __init__(self, vocab_size=300, **kwargs):
        super(Embedder, self).__init__()
        self.embedding_dim = kwargs["word_embedding_dim"]
        self.embedder = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
    
    def forward(self, x): #[batch_size, T, word_lst_num]
        return self.embedder(x.long())

class SelfAttention(nn.Module):
    def __init__(self, input_size, seq_len):
        """
        Args:
            input_size: int, hidden_size * num_directions
            seq_len: window_size
        """
        super(SelfAttention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.atten_bias.data.fill_(0)

    def forward(self, x):
        # x: [batch_size, window_size, 2*hidden_size]
        input_tensor = x.transpose(1, 0)  # w x b x h
        input_tensor = (torch.bmm(input_tensor, self.atten_w) + self.atten_bias)  # w x b x out
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()
        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), x).squeeze()
        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

class Trans(nn.Module):
    def __init__(self, input_size, layer_num, out_dim, dim_feedforward=512, dropout=0, device="cpu", norm=None, nhead=8):
        super(Trans, self).__init__()#default: 2048
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout, batch_first=True)
        encoder_norm = nn.LayerNorm(input_size)
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=layer_num, norm=encoder_norm).to(device)
        self.out_layer = nn.Linear(input_size, out_dim)
    def forward(self, x: torch.Tensor): #[batch_size, T, var]
        out = self.net(x)
        return self.out_layer(out)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,d_k=64):                      # Q: [batch_size, n_heads, len_q, d_k]
                                                                       # K: [batch_size, n_heads, len_k, d_k]
                                                                       # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                       # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)   # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)                           # 如果时停用词P就等于 0 
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                                # [batch_size, n_heads, len_q, d_v]
        return context, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model=512,n_heads=8,d_k='none',d_v='none',device="cpu"):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.d_k = d_k if d_k !='none' else int(d_model/n_heads)
        self.d_v = d_v if d_v !='none' else int(d_model/n_heads)
        self.W_Q = nn.Linear(d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * self.n_heads, bias=False)
        # self.fc = nn.Linear(n_heads * self.d_v, self.n_heads, bias=False)
        # self.fc = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, query, context, attn_mask='none'):    
        input_Q = query
        input_K = context
        input_V = query
                                                                # input_Q: [batch_size, len_q, d_model]
                                                                # input_K: [batch_size, len_k, d_model]
                                                                # input_V: [batch_size, len_v(=len_k), d_model]
                                                                # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  if attn_mask!='none' else torch.zeros(batch_size,self.n_heads,Q.size(2),K.size(2)).bool().to(self.device)           # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask,d_k=self.d_k)          # context: [batch_size, n_heads, len_q, d_v]
                                                                                 # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_v, n_heads * d_v]
        # output = self.fc(context)                                                # [batch_size, len_v, d_model]
        output = context
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn