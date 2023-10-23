import torch
import torch.nn as nn
from torch.nn.functional import softmax as sf
from models.utils import *
from models.kpi_model_v3 import KpiEncoder, KpiEncoder_low
from models.log_model_v3 import LogEncoder, LogEncoder_low
from models.utils import MultiHeadAttention
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
        context_vec = self.linear_in2(context.reshape(batch_size * output_len, dimensions))
        alpha=self.linear_out (self.tanh(query_vec+context_vec))
        alpha=alpha.reshape(batch_size,output_len,output_len)
        alpha=self.softmax(alpha)
        output= torch.bmm(alpha,context)
        output = torch.cat((output, context), dim=-1)
        output=self.fc(output)
        return output, alpha

class DotAttention(nn.Module): #k=V
    def __init__(self, dimensions):
        super(DotAttention, self).__init__()
        self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_in2 = nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context): #[batch_size, length, dim]
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        query = query.reshape(batch_size * output_len, dimensions)
        query = self.linear_in(query)
        query = query.reshape(batch_size, output_len, dimensions)
        # context = self.linear_in2(context)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())  # (batch_size, output_len, query_len)
        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)
        mix = torch.bmm(attention_weights, context) # (batch_size, output_len, dimensions)
        return mix, attention_weights

class MultiEncoder(nn.Module):
    def __init__(self, var_nums, device, vocab_size=300, fuse_type="cross_attn", **kwargs):
        super(MultiEncoder, self).__init__() 
        self.log_encoder = LogEncoder(device, **kwargs)
        self.kpi_encoder = KpiEncoder(device, **kwargs)
        self.hidden_size = kwargs["hidden_size"]  
        self.window_size = 100
        self.feature_type = kwargs["feature_type"]
        self.fuse_type = fuse_type
        if self.fuse_type == "cross_attn" or self.fuse_type == "sep_attn":
            if kwargs["attn_type"] == "add":
                self.attn_alpha = AddAttention(self.hidden_size,kwargs["window_size"])
                self.attn_beta = AddAttention(self.hidden_size,kwargs["window_size"])
            elif kwargs["attn_type"] == "qkv":
                self.attn_alpha = MultiHeadAttention(self.hidden_size,8,device=device)
                self.attn_beta = MultiHeadAttention(self.hidden_size,8,device=device)
            elif kwargs["attn_type"] == "dot":
                self.attn_alpha = DotAttention(self.hidden_size)
                self.attn_beta = DotAttention(self.hidden_size)
        elif  self.fuse_type == "multi_modal_self_attn":
            self.self_attention=MultiHeadAttention(2*self.hidden_size,2,device=device)
    def forward(self, log_x, kpi_x):
        kpi_re = self.kpi_encoder(kpi_x) #[batch_size, T, hidden_size]
        log_re = self.log_encoder(log_x) #[batch_size, W, hidden_size]
        fused = None
        if self.fuse_type == "cross_attn":
            fused_kpi, _ = self.attn_alpha(query=log_re, context=kpi_re)
            fused_log, _ = self.attn_beta(query=kpi_re, context=log_re)
            fused = torch.cat((fused_kpi, fused_log), dim=-1)
        elif self.fuse_type == "sep_attn":
            fused_kpi, _ = self.attn_alpha(query=kpi_re, context=kpi_re)
            fused_log, _ = self.attn_beta(query=log_re, context=log_re)
            fused = torch.cat((fused_kpi, fused_log), dim=-1)
        elif self.fuse_type == "concat":
            fused_kpi = kpi_re
            fused_log = log_re
            fused = torch.cat((kpi_re, log_re), dim=-1)
        elif self.fuse_type == "multi_modal_self_attn":
            fused_kpi = kpi_re
            fused_log = log_re
            fused = torch.cat((kpi_re, log_re), dim=-1)
            fused=self.self_attention(fused,fused)[0]
        return fused_kpi,fused_log,fused #[batch_size, T+W, hidden_size]
    
class ReturnSelf(nn.Module):
    def __init__(self):
        super(ReturnSelf, self).__init__()
        
    def forward(self,x):
        return x

class ReturnTopXWeight(nn.Module):
    def __init__(self):
        super(ReturnTopXWeight, self).__init__()
        self.relu = nn.ReLU()
        
    def forward(self,x):
        m = torch.quantile(x,0.8,dim=-1,keepdim=True)
        w = x - m
        return self.relu(w)+1.0
    
class Return0(nn.Module):
    def __init__(self):
        super(Return0, self).__init__()
        
    def forward(self,x):
        return 0
    
class Return1(nn.Module):
    def __init__(self):
        super(Return1, self).__init__()
        
    def forward(self,x):
        return 1

class Loss_fuse_model(nn.Module):
    def __init__(self,**keywds):
        super(Loss_fuse_model, self).__init__()
        self.weight = nn.Linear(keywds["window_size"]*2, keywds["window_size"], bias=False)
        if keywds["sigma_matrix"]:
            self.sigmas_dota=nn.Parameter(nn.init.uniform_(torch.empty(keywds["window_size"]),a=0.2,b=1.0),requires_grad=True)
        else: 
            self.sigmas_dota=nn.Parameter(nn.init.uniform_(torch.empty(1),a=0.2,b=1.0),requires_grad=True)
        self.weight2_0 = nn.Linear(keywds["hidden_size"]*2, 1, bias=False)
        self.s = ReturnSelf() if keywds["open_narrowing_modal_gap"] else Return0()
        self.f3 = ReturnTopXWeight() if keywds["open_expand_anomaly_gap"] else Return1()
         
    def forward(self,loss_set):
        w = self.sigmas_dota 
        log_d = loss_set[0]*self.f3(loss_set[0])
        kpi_d = loss_set[1]*self.f3(loss_set[1])
        loss_part = w*log_d + (1-w)*kpi_d + self.s(torch.abs(log_d-kpi_d)) # feature1
        return {"loss":loss_part,"w":w}
    
    def get_loss(self,res_set):
        log_x = res_set["features"][0]
        kpi_x = res_set["features"][1]
        log_d = res_set["distance"][0]*self.f3(res_set["distance"][0])
        kpi_d = res_set["distance"][1]*self.f3(res_set["distance"][1])
        w = self.weight2_0(torch.concatenate([kpi_x,log_x],dim=-1)).squeeze()
        loss_part = w*log_d + (1-w)*kpi_d + self.s(torch.abs(res_set["distance"][0]-res_set["distance"][1])) # feature1
        return {"loss":loss_part,"w":w} 
    
class MultiEncoder_low(nn.Module):
    def __init__(self, var_nums, device, vocab_size=300, fuse_type="cross_attn", **kwargs):
        super(MultiEncoder_low, self).__init__()
        self.log_encoder = LogEncoder_low(device, **kwargs).to(device)
        self.kpi_encoder = KpiEncoder_low(device, **kwargs).to(device)
        self.hidden_size = kwargs["hidden_size"]  
        self.window_size = 100
        self.feature_type = kwargs["feature_type"]
        self.fuse_type = "concat"
        # self.fuse_type = fuse_type
        if self.fuse_type == "cross_attn" or self.fuse_type == "sep_attn":
            if kwargs["attn_type"] == "add":
                self.attn_alpha = AddAttention(self.hidden_size)
                self.attn_beta = AddAttention(self.hidden_size)
            elif kwargs["attn_type"] == "qkv":
                self.attn_alpha = MultiHeadAttention(self.hidden_size,8,device=device)
                self.attn_beta = MultiHeadAttention(self.hidden_size,8,device=device)
            elif kwargs["attn_type"] == "dot":
                self.attn_alpha = DotAttention(self.hidden_size)
                self.attn_beta = DotAttention(self.hidden_size)
        elif  self.fuse_type == "multi_modal_self_attn":
            self.self_attention=MultiHeadAttention(2*self.hidden_size,2,device=device)
    def forward(self, log_x, kpi_x):
        kpi_re = self.kpi_encoder(kpi_x) #[batch_size, T, hidden_size]
        log_re = self.log_encoder(log_x) #[batch_size, W, hidden_size]
        # fuse_re = torch.cat((kpi_re, log_re), dim=-1)
        # return kpi_re,log_re,fuse_re
        
        fused = None
        if self.fuse_type == "cross_attn":
            fused_kpi, _ = self.attn_alpha(query=log_re, context=kpi_re)
            fused_log, _ = self.attn_beta(query=kpi_re, context=log_re)
            fused = torch.cat((fused_kpi, fused_log), dim=-1)
        elif self.fuse_type == "sep_attn":
            fused_kpi, _ = self.attn_alpha(query=kpi_re, context=kpi_re)
            fused_log, _ = self.attn_beta(query=log_re, context=log_re)
            fused = torch.cat((fused_kpi, fused_log), dim=-1)
        elif self.fuse_type == "concat":
            fused_kpi = kpi_re
            fused_log = log_re
            fused = torch.cat((kpi_re, log_re), dim=-1)
        elif self.fuse_type == "multi_modal_self_attn":
            fused_kpi = kpi_re
            fused_log = log_re
            fused = torch.cat((kpi_re, log_re), dim=-1)
            fused=self.self_attention(fused,fused)[0]
        return fused_kpi,fused_log,fused #[batch_size, T+W, hidden_size]
        
class MultiDiscriminator(nn.Module):
    def __init__(self, var_nums, device, fuse_type="cross_attn", **kwargs):
        super(MultiDiscriminator, self).__init__()
        self.fuse_type = fuse_type
        self.encoder = MultiEncoder_low(var_nums=var_nums, device=device, fuse_type=fuse_type, **kwargs)
        # self.encoder = MultiEncoder(var_nums=var_nums, device=device, fuse_type=fuse_type, **kwargs)
        self.fc = nn.Linear(kwargs["window_size"]*kwargs["hidden_size"]*2,kwargs["hidden_size"])
        self.decoder_fuse = nn.Linear(kwargs["hidden_size"],2)
        self.decoder = nn.Linear(kwargs["window_size"]*kwargs["hidden_size"]*2,2)
        self.decoder2 = nn.Linear(kwargs["window_size"]*kwargs["hidden_size"]*2,2)
        self.decoder3 = nn.Linear(kwargs["window_size"]*kwargs["hidden_size"],2)
        self.decoder4 = nn.Linear(kwargs["window_size"]*kwargs["hidden_size"],2)
        self.criterion = nn.CrossEntropyLoss() #nn.CrossEntropyLoss(reduction='none')
        self.criterion2 = nn.MSELoss()
    
    def get_loss_old(self, input_dict, res_set, flag=False):
        log_x = input_dict["log_features"]
        kpi_x = input_dict["kpi_features"]
        unmatched_kpi_x = input_dict["unmatched_kpi_features"]
        log_x_fake = res_set["output"][0]
        kpi_x_fake = res_set["output"][1]
        b,_,_ = kpi_x.shape
        kpi_re,log_re,concate_feature = self.encoder(log_x,kpi_x)
        pred = self.decoder(concate_feature.reshape(b,-1))
        kpi_re_fake,log_re_fake,concate_feature_fake = self.encoder(log_x_fake,kpi_x_fake)
        pred_fake = self.decoder(concate_feature_fake.reshape(b,-1))
        kpi_re_unmatched,log_re_unmatched,concate_feature_unmatched = self.encoder(log_x,unmatched_kpi_x)
        pred_unmatched = self.decoder(concate_feature_unmatched.reshape(b,-1))
        y1 = torch.ones_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        y2 = torch.zeros_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        # y3 = torch.ones_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device) * 2
        loss = self.criterion(pred,y1) + self.criterion(pred_fake,y2)

        deceive_loss = self.criterion2(concate_feature,concate_feature_fake)
        
        # if flag:
        #     loss += self.criterion(pred_unmatched,y2)
        #     deceive_loss =max(0,deceive_loss - self.criterion2(kpi_re_fake,kpi_re_unmatched)) # work
        #     # deceive_loss =max(0,deceive_loss - self.criterion2(concate_feature_fake,concate_feature_unmatched)) # prob work
        
        return {"loss":loss,"deceive_loss":deceive_loss}
    
    def get_loss_sep(self, input_dict, res_set, flag=False):
        log_x = input_dict["log_features"]
        kpi_x = input_dict["kpi_features"]
        unmatched_kpi_x = input_dict["unmatched_kpi_features"]
        log_x_fake = res_set["output"][0]
        kpi_x_fake = res_set["output"][1]
        b,_,_ = kpi_x.shape
        kpi_re,log_re,concate_feature_ = self.encoder(log_x,kpi_x)
        concate_feature = self.fc(concate_feature_.reshape(b,-1))
        pred = self.decoder_fuse(concate_feature)
        kpi_re_fake,log_re_fake,concate_feature_fake_ = self.encoder(log_x_fake,kpi_x_fake)
        concate_feature_fake = self.fc(concate_feature_fake_.reshape(b,-1))
        pred_fake = self.decoder_fuse(concate_feature_fake)
        kpi_re_unmatched,log_re_unmatched,concate_feature_unmatched_ = self.encoder(log_x,unmatched_kpi_x)
        concate_feature_unmatched = self.fc(concate_feature_unmatched_.reshape(b,-1))
        pred_unmatched = self.decoder_fuse(concate_feature_unmatched)
        y1 = torch.ones_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        y2 = torch.zeros_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        # y3 = torch.ones_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device) * 2
        loss = self.criterion(pred,y1) + self.criterion(pred_fake,y2)

        deceive_loss = self.criterion2(concate_feature,concate_feature_fake)
        # if flag:
        #     loss += self.criterion(pred_unmatched,y2)
        #     deceive_loss =max(0,deceive_loss - self.criterion2(concate_feature_fake,concate_feature_unmatched)+0.1) # prob work
        #     # deceive_loss =max(0,deceive_loss - self.criterion2(kpi_re_fake,kpi_re_unmatched)) # work
        return {"loss":loss,"deceive_loss":deceive_loss}

    
    def get_loss(self, input_dict, res_set, flag=False):
        log_x = input_dict["log_features"]
        kpi_x = input_dict["kpi_features"]
        # unmatched_kpi_x = input_dict["unmatched_kpi_features"]
        log_x_fake = res_set["output"][0]
        kpi_x_fake = res_set["output"][1]
        b,_,_ = kpi_x.shape
        kpi_re,log_re,concate_feature = self.encoder(log_x,kpi_x)
        pred = self.decoder2(concate_feature.reshape(b,-1))
        pred_log = self.decoder3(log_re.reshape(b,-1))
        pred_kpi = self.decoder4(kpi_re.reshape(b,-1))  
        kpi_re_fake,log_re_fake,concate_feature_fake = self.encoder(log_x_fake,kpi_x_fake)
        pred_fake = self.decoder2(concate_feature_fake.reshape(b,-1))
        pred_log_fake = self.decoder3(log_re_fake.reshape(b,-1))
        pred_kpi_fake = self.decoder4(kpi_re_fake.reshape(b,-1))
        # kpi_re_unmatched,log_re_unmatched,concate_feature_unmatched = self.encoder(log_x,unmatched_kpi_x)
        # pred_unmatched = self.decoder2(concate_feature_unmatched.reshape(b,-1))
        # pred_log_unmatched = self.decoder3(log_re_unmatched.reshape(b,-1))
        # pred_kpi_unmatched = self.decoder4(kpi_re_unmatched.reshape(b,-1))
        
        y1 = torch.ones_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        y2 = torch.zeros_like(pred).mean(dim=-1).type(torch.LongTensor).to(pred.device)
        loss = self.criterion(pred_kpi_fake,y2)  + self.criterion(pred_kpi,y1) + self.criterion(pred_log,y1) + self.criterion(pred_log_fake,y2)
        deceive_loss = self.criterion2(kpi_re,kpi_re_fake)  + self.criterion2(log_re,log_re_fake)
        # if flag:
        #     loss += self.criterion(pred_kpi_unmatched,y2)
        #     deceive_loss = max(0, deceive_loss - self.criterion2(kpi_re_fake,kpi_re_unmatched)+0.2)
        return {"loss":loss,"deceive_loss":deceive_loss}

class MultiModel(nn.Module):
    def __init__(self, var_nums, device, fuse_type="cross_attn", **kwargs):
        super(MultiModel, self).__init__()
        self.fuse_type = fuse_type
        self.hidden_size = kwargs["hidden_size"]
        self.encoder = MultiEncoder(var_nums=var_nums, device=device, fuse_type=fuse_type, **kwargs)
        self.log_c = kwargs["log_c"]
        self.kpi_c = kwargs["kpi_c"]
        # self.unmatch_k = kwargs["unmatch_k"]*0.1 + 1.0 # 之前的实验
        self.unmatch_k = kwargs["unmatch_k"]*0.01  # 之前的实验
        self.fuse_decoder=nn.Linear(kwargs["hidden_size"]*2,self.kpi_c+self.log_c)
        if kwargs["criterion"] == "l1":
            self.criterion1 = nn.L1Loss()
            self.criterion2 = nn.L1Loss(reduction='none')
        else:
            self.criterion1 = nn.MSELoss()
            self.criterion2 = nn.MSELoss(reduction='none')
        
        
        self.criterion3 = nn.MSELoss()
        self.criterion4 = nn.L1Loss()
        self.narrow_modal_gap = ReturnSelf() if kwargs["open_narrowing_modal_gap"] else Return0()
        self.expand_anomaly_gap = ReturnTopXWeight() if kwargs["open_expand_anomaly_gap"] else Return1()

    def forward(self, input_dict, flag=False):
        input = torch.concatenate([input_dict["kpi_features"],input_dict["log_features"]],dim=-1)
        input_unmatched = torch.concatenate([input_dict["unmatched_kpi_features"],input_dict["log_features"]],dim=-1)
        fused_kpi,fused_log,concat_feature = self.encoder(input_dict["log_features"],input_dict["kpi_features"])
        fused_kpi_unmatched,fused_log_unmatched,concat_feature_unmatched = self.encoder(input_dict["log_features"],input_dict["unmatched_kpi_features"])
        fused_out_unmatched = self.fuse_decoder(concat_feature_unmatched)
        fused_out = self.fuse_decoder(concat_feature)
        kpi_out=fused_out[:,:,:self.kpi_c]
        log_out =fused_out[:,:,self.kpi_c:]
        kpi_out_unmatched=fused_out_unmatched[:,:,:self.kpi_c]
        log_out_unmatched =fused_out_unmatched[:,:,self.kpi_c:]
        fused_kpi_fake,fused_log_fake,concat_feature_fake = self.encoder(log_out,kpi_out)
        fused_out_fake = self.fuse_decoder(concat_feature_fake)
        
        kpi_dis=self.criterion2(kpi_out,input_dict["kpi_features"]).mean(dim=-1)
        log_dis=self.criterion2(log_out,input_dict["log_features"]).mean(dim=-1)
        

        # log_d = log_dis
        # kpi_d = kpi_dis
        log_d = log_dis* self.expand_anomaly_gap(log_dis)
        kpi_d = kpi_dis* self.expand_anomaly_gap(kpi_dis)
        fusion_loss = log_d + kpi_d + self.narrow_modal_gap(torch.abs(log_d-kpi_d))
        # loss = log_dis.mean() + kpi_dis.mean()+ max(0,self.criterion3(concat_feature,concat_feature_fake) - self.criterion3(concat_feature,concat_feature_unmatched))
        
        loss = log_dis.mean() + kpi_dis.mean()
        # loss = self.criterion4(input,fused_out)
        
        if flag:
            # loss += max(0,self.criterion3(concat_feature,concat_feature_fake) - self.criterion3(concat_feature[:,:,:self.hidden_size],concat_feature_unmatched[:,:,:self.hidden_size])-0.1)
            # loss += max(0,self.criterion4(input[:,:,:self.kpi_c],input_unmatched[:,:,:self.kpi_c])*1.2-self.criterion4(fused_out_fake[:,:,:self.kpi_c],fused_out_unmatched[:,:,:self.kpi_c]))
            # loss += max(0,self.criterion4(input,input_unmatched)*1.5-self.criterion4(fused_out_fake,fused_out_unmatched))
            # loss += max(0,self.criterion4(input,fused_out)*1.2-self.criterion4(input_unmatched,fused_out_unmatched))
            # loss += max(0,self.criterion4(input_dict["kpi_features"],kpi_out)*self.unmatch_k-self.criterion4(input_dict["unmatched_kpi_features"],kpi_out_unmatched)) # 之前的实验
            loss += max(0,self.criterion4(input_dict["kpi_features"],kpi_out)+ self.unmatch_k -self.criterion4(input_dict["unmatched_kpi_features"],kpi_out_unmatched))


            # loss += self.criterion3(concat_feature,concat_feature_fake) + max(0,self.criterion3(concat_feature,concat_feature_fake)*2 - self.criterion3(concat_feature[:,:,:self.hidden_size],concat_feature_unmatched[:,:,:self.hidden_size]))
            # loss = log_dis.mean() + kpi_dis.mean()+ max(0, 0.1 - self.criterion3(concat_feature,concat_feature_unmatched))
        
        return {"fusion_loss":fusion_loss,"loss":loss,"dis":(log_dis,kpi_dis),"features":(fused_log,fused_kpi,concat_feature), "output":(log_out,kpi_out)}
