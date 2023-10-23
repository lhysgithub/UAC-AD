import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from models.utils import SelfAttention, Embedder, Decoder, Trans
from torch.nn.functional import softmax as sf
import math
from math import sqrt

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
    
class LogEncoder(nn.Module):
    def __init__(self, device, log_dropout=0, log_layer_num=4, transformer_hidden=512, **kwargs):
        super(LogEncoder, self).__init__()
        padding=1 if torch.__version__>="1.5.0" else 2
        self.net = Trans(input_size=kwargs["hidden_size"], layer_num=log_layer_num, out_dim=kwargs["hidden_size"],
                dim_feedforward=transformer_hidden, dropout=log_dropout, device=device)
        self.token_embed=nn.Conv1d(in_channels=kwargs["log_c"],out_channels=kwargs["hidden_size"],kernel_size=3,padding_mode="circular",padding=padding)
        self.position_embed = PositionalEmbedding(kwargs["hidden_size"]) if kwargs["open_position_embedding"] else Return0()
    
    def forward(self, session: torch.Tensor): #session: [batch_size,  seq_lenth, embedding_dim]
        session_emb=self.token_embed(session.permute(0,2,1)).permute(0,2,1) + self.position_embed(session)
        log_re=self.net(session_emb)
        return log_re #[batch_size, window_size, hidden_size]

class LogEncoder_low(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(LogEncoder_low,self).__init__()
        padding=1 if torch.__version__>="1.5.0" else 2
        self.token_embed=nn.Conv1d(in_channels=kwargs["log_c"],out_channels=kwargs["hidden_size"],kernel_size=3,padding_mode="circular",padding=padding)
        self.position_embed = PositionalEmbedding(kwargs["hidden_size"]) if kwargs["open_position_embedding"] else Return0()
    def forward(self, session: torch.Tensor): #session: [batch_size,  seq_lenth, embedding_dim]
        session_emb = self.token_embed(session.permute(0,2,1)).permute(0,2,1)  + self.position_embed(session)
        # kpi_re = self.net(session_emb)
        return session_emb #[batch_size, window_size, hidden_size]

class LogDiscriminator(nn.Module):
    def __init__(self, device, vocab_size=300, **kwargs):
        super(LogDiscriminator, self).__init__()
        self.encoder = LogEncoder_low(device, **kwargs)
        self.decoder=nn.Linear(kwargs["hidden_size"]*kwargs["window_size"],2)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion2 = nn.MSELoss()

    def get_loss(self, input, res, flag=False):
        log_x = input["log_features"]
        log_x_fake = res["output"]
        b,_,_ = log_x.shape
        log_re = self.encoder(log_x) #[batch_size, W, hidden_size*dir_num]
        pred = self.decoder(log_re.reshape(b,-1))
        log_re_fake = self.encoder(log_x_fake)
        pred_fake = self.decoder(log_re_fake.reshape(b,-1))
        y1 = torch.ones_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        y2 = torch.zeros_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        loss = self.criterion(pred,y1) + self.criterion(pred_fake,y2)
        # deceive_loss = self.criterion(pred_fake,y1)
        deceive_loss = self.criterion2(log_re,log_re_fake)
        return {"loss":loss,"deceive_loss":deceive_loss}
        

class LogModel(nn.Module):
    def __init__(self, device, vocab_size=300, **kwargs):
        super(LogModel, self).__init__()
        self.encoder = LogEncoder(device, **kwargs)
        self.decoder=nn.Linear(kwargs["hidden_size"],kwargs["log_c"])
        if kwargs["criterion"] == "l1":
            self.criterion1 = nn.L1Loss()
            self.criterion2 = nn.L1Loss(reduction='none')
        else:
            self.criterion1 = nn.MSELoss()
            self.criterion2 = nn.MSELoss(reduction='none')

    def forward(self, input, flag=False):
        log_x = input["log_features"]
        log_re = self.encoder(log_x) #[batch_size, W, hidden_size*dir_num]
        logits = self.decoder(log_re)
        distance=self.criterion2(logits,log_x).mean(dim=-1)
        return {"loss": distance,"output":logits}