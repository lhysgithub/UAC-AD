import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from models.utils import SelfAttention, Embedder, Decoder, Trans
from torch.nn.functional import softmax as sf
import math
from math import sqrt


class AddAttention(nn.Module): #k=V
    def __init__(self, dimensions,windows_lens=100):
        super(AddAttention, self).__init__()
        self.linear_in = nn.Linear(dimensions, dimensions, bias=True)
        self.linear_in2 = nn.Linear(dimensions, dimensions, bias=True)

        self.linear_out = nn.Linear(dimensions, windows_lens, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.fc=nn.Linear(2*dimensions, dimensions, bias=True)

    def forward(self, query, context): #[batch_size, length, dim]
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        query_vec = self.linear_in(query.reshape(batch_size * output_len, dimensions))
        # query = query.reshape(batch_size, output_len, dimensions)
        context_vec = self.linear_in2(context.reshape(batch_size * output_len, dimensions))
        alpha=self.linear_out (self.tanh(query_vec+context_vec))
        alpha=alpha.reshape(batch_size,output_len,output_len)
        alpha=self.softmax(alpha)
        output= torch.bmm(alpha,context)
        output = torch.cat((output, context), dim=-1)
        output=self.fc(output)

        return output, alpha


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_sizes, dilation=2, device="cpu", dropout=0, pooling=True, **kwargs):
        super(ConvNet, self).__init__()
        layers = []
        for i in range(len(kernel_sizes)):
            dilation_size = dilation ** i
            kernel_size = kernel_sizes[i]
            padding = (kernel_size-1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding), nn.ReLU(), Chomp1d(padding), nn.Dropout(dropout)]
            
        self.network = nn.Sequential(*layers)
        
        self.pooling = pooling
        if self.pooling:
            self.maxpool = nn.MaxPool1d(num_channels[-1]).to(device)
        self.network.to(device)
        
    
    def forward(self, x): #[batch_size, T, 1]
        x = x.permute(0, 2, 1) #[batch_size, 1, T]
        out = self.network(x) #[batch_size, out_dim, T]
        out = out.permute(0, 2, 1) #[batch_size, T, out_dim]
        if self.pooling:
            return self.maxpool(out)
        else:
            return out

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class Return0(nn.Module):
    def __init__(self,):
        super(Return0, self).__init__()

    def forward(self, x):
        return 0


class TemporalEncoder(nn.Module):
    def __init__(self, device, input_size, **kwargs):
        super(TemporalEncoder, self).__init__()
        hidden_sizes = kwargs["temporal_hidden_sizes"]
        kernel_sizes = kwargs["temporal_kernel_sizes"]
        dropout = kwargs["temporal_dropout"]
        pooling = kwargs["pooling"]
        self.temporal_dim = hidden_sizes[-1]
         
        self.temporal_dim = 1 if pooling else hidden_sizes[-1]
        self.net = ConvNet(input_size, hidden_sizes, kernel_sizes, device=device, dropout=dropout, pooling=pooling)

    def forward(self, x: torch.Tensor): #[batch_size, T, input_size] --> [batch_size, T, temporal_dim]
        x = x.type("torch.FloatTensor").to(x.device)
        return self.net(x) 

class KpiEncoder_low(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(KpiEncoder_low, self).__init__()
        padding=1 if torch.__version__>="1.5.0" else 2
        self.token_embed=nn.Conv1d(in_channels=kwargs["kpi_c"],out_channels=kwargs["hidden_size"],kernel_size=3,padding_mode="circular",padding=padding)
        self.position_embed = PositionalEmbedding(kwargs["hidden_size"]) if kwargs["open_position_embedding"] else Return0()
    def forward(self, session: torch.Tensor): #session: [batch_size,  seq_lenth, embedding_dim]
        session_emb = self.token_embed(session.permute(0,2,1)).permute(0,2,1)  + self.position_embed(session)
        # kpi_re = self.net(session_emb)
        return session_emb #[batch_size, window_size, hidden_size]

class KpiDiscriminator(nn.Module):
    def __init__(self, device, **kwargs):
        super(KpiDiscriminator, self).__init__()
        self.encoder = KpiEncoder_low(device, **kwargs)
        self.decoder=nn.Linear(kwargs["window_size"]*kwargs["hidden_size"],2)
        self.decoder2 = nn.Linear(kwargs["window_size"]*kwargs["kpi_c"],2)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion2 = nn.MSELoss()

    def get_loss(self, input, res, flag=False):
        kpi_x = input["kpi_features"]
        # unmatched_kpi_x = input["unmatched_kpi_features"]
        kpi_x_fake = res["output"]
        b,_,_ = kpi_x.shape
        
        kpi_re = self.encoder(kpi_x) #[batch_size, W, hidden_size*dir_num]
        pred = self.decoder(kpi_re.reshape(b,-1))
        kpi_re_fake = self.encoder(kpi_x_fake)
        pred_fake = self.decoder(kpi_re_fake.reshape(b,-1))
        # attend_kpi, _ = self.attn_alpha(query=kpi_re, context=kpi_re)
        
        # pred = self.decoder2(kpi_x.reshape(b,-1))
        # pred_fake = self.decoder2(kpi_x_fake.reshape(b,-1))
        # kpi_re_unmatched = self.encoder(unmatched_kpi_x)
        # pred_unmatched = self.decoder(kpi_re_unmatched.reshape(b,-1))
        y1 = torch.ones_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        y2 = torch.zeros_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        # y3 = torch.ones_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device) * 2
        loss = self.criterion(pred,y1) + self.criterion(pred_fake,y2) # + self.criterion(pred_unmatched,y3)
        # deceive_loss = self.criterion(pred_fake,y1)
        deceive_loss = self.criterion2(kpi_re,kpi_re_fake)
        return {"loss":loss,"deceive_loss":deceive_loss}

class KpiEncoder(nn.Module):
    def __init__(self, device, inner_dropout=0, kpi_layer_num=4, transformer_hidden=512, **kwargs):
        super(KpiEncoder, self).__init__()
        padding=1 if torch.__version__>="1.5.0" else 2
        self.net = Trans(input_size=kwargs["hidden_size"], layer_num=kpi_layer_num, out_dim=kwargs["hidden_size"],
                dim_feedforward=transformer_hidden, dropout=inner_dropout, device=device)
        self.token_embed=nn.Conv1d(in_channels=kwargs["kpi_c"],out_channels=kwargs["hidden_size"],kernel_size=3,padding_mode="circular",padding=padding)
        self.position_embed = PositionalEmbedding(kwargs["hidden_size"]) if kwargs["open_position_embedding"] else Return0()
        self.attn_alpha = AddAttention(kwargs["hidden_size"])
    
    def forward(self, session: torch.Tensor): #session: [batch_size,  seq_lenth, embedding_dim]
        session_emb = self.token_embed(session.permute(0,2,1)).permute(0,2,1)  + self.position_embed(session)
        kpi_re = self.net(session_emb)
        return kpi_re #[batch_size, window_size, hidden_size]

class KpiModel(nn.Module):
    def __init__(self, device, **kwargs):
        super(KpiModel, self).__init__()
        self.encoder = KpiEncoder(device, **kwargs)
        self.decoder=nn.Linear(kwargs["hidden_size"],kwargs["kpi_c"])
        if kwargs["criterion"] == "l1":
            self.criterion1 = nn.L1Loss()
            self.criterion2 = nn.L1Loss(reduction='none')
        else:
            self.criterion1 = nn.MSELoss()
            self.criterion2 = nn.MSELoss(reduction='none')

    def forward(self, input, flag=False):
        kpi_x = input["kpi_features"]
        kpi_re = self.encoder(kpi_x) #[batch_size, W, hidden_size*dir_num]
        # attend_kpi, _ = self.attn_alpha(query=kpi_re, context=kpi_re)
        logits = self.decoder(kpi_re)
        distance=self.criterion2(logits,kpi_x).mean(dim=-1)
        return {"loss": distance,"output":logits}