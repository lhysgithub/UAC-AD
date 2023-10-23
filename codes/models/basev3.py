import os
import time
import copy
from collections import defaultdict
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, recall_score, precision_score
import logging
import warnings
warnings.filterwarnings("ignore", module="sklearn")
from sklearn.cluster import KMeans
import numpy as np
from models.fuse_v3 import MultiModel, Loss_fuse_model, MultiDiscriminator
from models.log_model_v3 import LogModel, LogDiscriminator
from models.kpi_model_v3 import KpiModel, KpiDiscriminator
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from models.utils import *
import matplotlib.pyplot as plt
from thop import profile
from sklearn.preprocessing import MinMaxScaler, RobustScaler
def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
        # scaler.data_range_
    data = scaler.transform(data)
    # print("Data normalized")

    return data, scaler


def find_positive_segment(data):
    if isinstance(data,str):
        return [],[]
    dataset_len = int(len(data))
    index_list = []
    lens_list = []
    find = 0
    count = 0
    for i in range(len(data)):
        if int(data[i]) == 1:
            if find == 0:
                index_list.append(i)
            find = 1
            count += 1
        elif find == 1:
            find = 0
            lens_list.append(count)
            count = 0
    if find == 1:
        find = 0
        lens_list.append(count)
        count = 0
    return index_list, lens_list


def plot_curve(data,name):
    l , k = data.shape
    for i in range(k):
        plt.cla()
        x = range(len(data))
        plt.plot(x, data[:,i], "r", label=f"{name}_{i}",alpha=0.5)
        plt.legend()
        plt.savefig(f"temp/{name}_{i}.pdf")


def plot_labeled_curve(data,labels,name):
    l , k = data.shape
    index_list, lens_list = find_positive_segment(labels)
    for i in range(k):
        if i !=106:
            continue
        plt.cla()
        x = range(len(data))
        plt.plot(x, data[:,i], "b", label=f"{name}_{i}",alpha=0.5)
        for j in range(len(index_list)):
            start = index_list[j]
            end = index_list[j] + lens_list[j]
            y_min = data[:,i].min()
            y_max = data[:,i].max()
            plt.fill_between([start, end], y_min, y_max, facecolor='pink', alpha=0.8)
        plt.legend()
        plt.savefig(f"temp/{name}_{i}.pdf")


def plot_labeled_2curve(data,data2,labels,labels2,name):
    l , k = data.shape
    index_list, lens_list = find_positive_segment(labels)
    index_list2, lens_list2 = find_positive_segment(labels2)
    for i in range(k):
        # if i !=106:
        #     continue
        # if i %100 !=0:
        #     continue
        plt.cla()
        x = range(len(data))
        plt.plot(x, data[:,i], "b", label=f"pred_{i}",alpha=0.5)
        plt.plot(x, data2[:,i], "k", label=f"gt_{i}",alpha=0.5)
        for j in range(len(index_list)):
            start = index_list[j]
            end = index_list[j] + lens_list[j]
            y_min = data[:,i].min()
            y_max = data[:,i].max()
            plt.fill_between([start, end], y_min, y_max, facecolor='green', alpha=0.8)
        for j in range(len(index_list2)):
            start = index_list2[j]
            end = index_list2[j] + lens_list2[j]
            y_min = data[:,i].min()
            y_max = data[:,i].max()
            plt.fill_between([start, end], y_min, y_max, facecolor='pink', alpha=0.8)
        plt.legend()
        plt.savefig(f"temp/{i}_{name}.pdf")

# from run import normalize_data
class pseudo_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def get_best_f1_pred(distance,gt,ratios):
    best_res = {"f1":0}
    best_pred = []
    for ano in ratios:
        threshold= np.percentile(distance,100-ano)
        pred=[1 if d>threshold  else 0 for d in distance]
        pred = adjustment_decision(pred,gt)  # 增加adjustment_decision
        eval_results = {
                    "f1": f1_score(gt, pred),
                    "rc": recall_score(gt, pred),
                    "pc": precision_score(gt, pred),
                }

        if eval_results["f1"] > best_res["f1"]:
                best_res = eval_results
                best_pred = pred
    return best_res,best_pred

class BaseModel(nn.Module):
    def __init__(self, device, var_nums, vocab_size=300, data_type="fuse", **kwargs):
        super(BaseModel, self).__init__()
        # Training
        [self.epoches_1, self.epoches_2] = kwargs["epoches"]
        self.confidence = kwargs["confidence"]
        self.alpha = kwargs["alpha"]
        self.batch_size = kwargs["batch_size"]
        self.learning_rate = kwargs["learning_rate"]
        self.patience = kwargs["patience"] # > 0: use early stop
        self.device = device
        self.var_nums = var_nums
        self.data_type = data_type
        self.current_epoch = 0
        self.open_min_max = kwargs["open_min_max"]
        self.k = kwargs["k"]
        self.openfeature2 = kwargs["open_feature2"]
        self.evaluation_sep = kwargs["evaluation_sep"]
        self.open_gan = kwargs["open_gan"]
        self.open_gan_sep = kwargs["open_gan_sep"]
        self.open_unmatch_zoomout = kwargs["open_unmatch_zoomout"]
        self.log_c = kwargs["log_c"]
        self.kpi_c = kwargs["kpi_c"]
        self.hidden_size = kwargs["hidden_size"]
        self.anomaly_rate = kwargs["anomaly_rate"]
        self.kwargs = kwargs

        self.model_save_dir = os.path.join(kwargs["result_dir"], kwargs["hash_id"])
        self.model_save_file = os.path.join(self.model_save_dir, 'model.ckpt')
        if data_type == "fuse":
            self.model = MultiModel(var_nums=var_nums, vocab_size=vocab_size, device=device, **kwargs).to(device)
            self.discriminator=MultiDiscriminator(var_nums=var_nums, vocab_size=vocab_size, device=device, **kwargs).to(device)
        elif data_type == "log":
            self.model = LogModel(vocab_size=vocab_size, device=device, **kwargs).to(device)
            self.discriminator=LogDiscriminator(vocab_size=vocab_size, device=device, **kwargs).to(device)
        elif data_type == "kpi":
            self.model = KpiModel(device=device, **kwargs).to(device) # 
            self.discriminator=KpiDiscriminator(device=device, **kwargs).to(device)
        self.loss_fusion=Loss_fuse_model(**kwargs).to(device)
        self.train_time = []

    def __input2device(self, batch_input):
        res = {}
        for key, value in batch_input.items():
            if isinstance(value, list):
                res[key] = [v.to(self.device) for v in value]
            else:
                res[key] = value.to(self.device)
        return res

    def load_model(self, model_save_file=""):
        self.model.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def save_model(self, state):
        try:
            torch.save(state, self.model_save_file, _use_new_zipfile_serialization=False)
        except:
            torch.save(state, self.model_save_file)

    def inference(self, data_loader):
        self.model.eval()
        data = []
        inference_time = []
        with torch.no_grad():
            for _input in data_loader:
                infer_start = time.time()
                result = self.model.forward(self.__input2device(_input), flag=True)
                inference_time.append(time.time() - infer_start)
                if result["conf"][0] >= self.confidence:
                    sample = {
                        "idx": _input["idx"].item(),
                        "label": int(result["y_pred"][0]),
                        "kpi_features": [ts.squeeze() for ts in _input["kpi_features"]],
                        "log_features": _input["log_features"].squeeze()
                    }
                    data.append(sample)
        logging.info("Inference delay {:.4f}".format(np.mean(inference_time)))
        return pseudo_Dataset(data)

    def evaluate(self, test_loader, datatype="Test"):
        res = defaultdict(list)
        kpi_inputs = []
        log_inputs = []
        kpi_outputs = []
        log_outputs = []
        embeds = []
        self.model.eval()
        self.loss_fusion.eval()
        with torch.no_grad():
            batch_cnt = 0
            for batch_input in test_loader:
                batch_cnt += 1
                kpi_inputs.append(batch_input["kpi_features"])
                log_inputs.append(batch_input["log_features"])
                result = self.model.forward(self.__input2device(batch_input))
                if self.data_type == "fuse":
                    distance = result["fusion_loss"]
                    embeds.append(result["features"][2].cpu().numpy())
                    kpi_outputs.append(result["output"][1].cpu().numpy())
                    log_outputs.append(result["output"][0].cpu().numpy())
                elif self.data_type == "kpi":
                    kpi_outputs.append(result["output"].cpu().numpy())
                    log_outputs.append(log_inputs[-1])
                    distance = result["loss"]
                else:
                    kpi_outputs.append(kpi_inputs[-1])
                    log_outputs.append(result["output"].cpu().numpy())
                    distance = result["loss"]
                res["loss"].extend(distance.cpu().numpy().reshape(-1).tolist())
                res["true"].extend(batch_input["labels"].cpu().numpy().reshape(-1).tolist())
        
        kpi_inputs = np.concatenate(kpi_inputs,axis=0).reshape(-1,self.kpi_c)
        kpi_outputs = np.concatenate(kpi_outputs,axis=0).reshape(-1,self.kpi_c)
        log_inputs = np.concatenate(log_inputs,axis=0).reshape(-1,self.log_c)
        log_outputs = np.concatenate(log_outputs,axis=0).reshape(-1,self.log_c)
        # inputs = np.concatenate([kpi_inputs,log_inputs],axis=-1)
        # outputs = np.concatenate([kpi_outputs,log_outputs],axis=-1)
        # embeds = np.concatenate(embeds,axis=0).reshape(-1,self.hidden_size*2)
        test_embeds = {"embeds":embeds,"distance":res["loss"],"labels":res["true"]}
        
        # anomaly_ratio= range(1, 101) # [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,]
        anomaly_ratio= range(1, self.anomaly_rate)
        best_res = {"f1":-1}
        best_pred = []
        for ano in anomaly_ratio:
            threshold= np.percentile(res["loss"],100-ano)
            pred=[1 if d>threshold  else 0 for d in res["loss"]]
            pred = adjustment_decision(pred,res["true"])  # 增加adjustment_decision
            eval_results = {
                    "f1": f1_score(res["true"], pred),
                    "rc": recall_score(res["true"], pred),
                    "pc": precision_score(res["true"], pred),
                }
            if eval_results["f1"] > best_res["f1"]:
                best_res = eval_results
                best_pred = pred
                
        # plot_labeled_2curve(kpi_outputs,kpi_inputs,best_pred,res["true"],f"test_kpi_pred_gt_label_{self.current_epoch}")
        # plot_labeled_2curve(log_outputs,log_inputs,best_pred,res["true"],f"test_log_pred_gt_label_{self.current_epoch}")
        return best_res,test_embeds

    def evaluate_sep(self, test_loader, datatype="Test"):
        self.model.eval()
        res = defaultdict(list)
        self.discriminator.eval()
        with torch.no_grad():
            batch_cnt = 0
            for batch_input in test_loader:
                batch_cnt += 1
                result = self.model.forward(self.__input2device(batch_input))
                res["distance1"].extend(result["dis"][0].cpu().numpy().reshape(-1).tolist())
                res["distance2"].extend(result["dis"][1].cpu().numpy().reshape(-1).tolist())
                res["true"].extend(batch_input["labels"].cpu().numpy().reshape(-1).tolist())
        # anomaly_ratio= range(1, 101) # [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,]
        anomaly_ratio= range(1, 5)
        
        log_best_res,log_best_pred = get_best_f1_pred(res["distance1"],res["true"],anomaly_ratio)
        kpi_best_res,kpi_best_pred = get_best_f1_pred(res["distance2"],res["true"],anomaly_ratio)
        multi_pred = np.array([log_best_pred,kpi_best_pred]).max(axis = 0) # 并
        # multi_pred = np.array([log_best_pred,kpi_best_pred]).min(axis = 0) # 交
        results = {
                    "f1": f1_score(res["true"], multi_pred),
                    "rc": recall_score(res["true"], multi_pred),
                    "pc": precision_score(res["true"], multi_pred),
                }
        
        return results

    def supervised_fit(self, train_loader, test_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_f1 = -1
        best_state, best_test_scores = None, None

        pre_loss, worse_count = float("inf"), 0
        for epoch in range(1, self.epoches_1+1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()
            for batch_input in train_loader:
                optimizer.zero_grad()
                loss = self.model.forward(self.__input2device(batch_input))["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt

            self.train_time.append(epoch_time_elapsed)
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoches_1, epoch_loss, epoch_time_elapsed))

            #train_results = self.evaluate(train_loader, datatype="Train")
            test_results = self.evaluate(test_loader, datatype="Test")
                

            if test_results["f1"] > best_f1:
                best_f1 = test_results["f1"]
                best_test_scores = test_results
                best_state = copy.deepcopy(self.model.state_dict())

        self.save_model(best_state)
        self.load_model(self.model_save_file)
        test_results = self.evaluate(test_loader, datatype="Test")
        logging.info("*** Test F1 {:.4f}  of supervised traning".format(test_results["f1"]))

        return best_test_scores

    def fit(self, train_loader, unlabel_loader, test_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_f1 = -1
        best_state, best_test_scores = None, None

        ##############################################################################
        ####                    Training with labeled data                        ####
        ##############################################################################
        pre_loss, worse_count = float("inf"), 0
        for epoch in range(1, self.epoches_1+1):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()
            for batch_input in train_loader:
                optimizer.zero_grad()
                loss = self.model.forward(self.__input2device(batch_input))["loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt

            self.train_time.append(epoch_time_elapsed)
            logging.info("Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, self.epoches_1, epoch_loss, epoch_time_elapsed))

            train_results = self.evaluate(train_loader, datatype="Train")
            test_results = self.evaluate(test_loader, datatype="Test")

            if test_results["f1"] > best_f1:
                best_f1 = test_results["f1"]
                best_test_scores = test_results
                best_state = copy.deepcopy(self.model.state_dict())

        self.save_model(best_state)
        self.load_model(self.model_save_file)
        test_results = self.evaluate(test_loader, datatype="Test")
        logging.info("*** Test F1 {:.4f}  of traning phase 1".format(test_results["f1"]))

        ##############################################################################
        ####              Training with labeled data and pseudo data              ####
        ##############################################################################
        pseudo_data = self.inference(unlabel_loader)
        pseudo_loader = DataLoader(pseudo_data, batch_size=self.batch_size, shuffle=True)

        pre_loss, worse_count = float("inf"), 0
        phase = False
        for epoch in range(1, self.epoches_2):
            self.model.train()
            batch_cnt, epoch_loss = 0, 0.0
            epoch_time_start = time.time()

            train_iterator = iter(train_loader)
            for pseudo_input in pseudo_loader:
                try:
                    train_input = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    train_input = next(train_iterator)

                optimizer.zero_grad()
                loss_1 = self.model.forward(self.__input2device(train_input))["loss"]
                loss_2 = self.model.forward(self.__input2device(pseudo_input))["loss"]
                loss = (1-self.alpha)*loss_1+self.alpha*loss_2
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1

            epoch_time_elapsed = time.time() - epoch_time_start
            epoch_loss = epoch_loss / batch_cnt
            self.train_time.append(epoch_time_elapsed)
            logging.info("Epoch {}/{}, training loss with real & pseudo labels: {:.5f} [{:.2f}s]".format(epoch, self.epoches_2, epoch_loss, epoch_time_elapsed))

            test_results = self.evaluate(test_loader, datatype="Test")

            if test_results["f1"] > best_f1:
                best_f1 = test_results["f1"]
                best_test_scores = test_results
                best_state = copy.deepcopy(self.model.state_dict())
                phase = True

            if epoch_loss > pre_loss:
                worse_count += 1
                if self.patience > 0 and worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break
            else: worse_count = 0
            pre_loss = epoch_loss

        self.save_model(best_state)
        self.load_model(self.model_save_file)
        test_results = self.evaluate(test_loader, datatype="Test")
        if phase:
            logging.info("*** Test F1 {:.4f} of traning phase 2".format(test_results["f1"]))
        else:
            logging.info("---- Training Phase 2 has no benifit!")

        logging.info("Best f1: test f1 {:.4f}".format(best_f1))
        return best_test_scores

    def unsupervised_fit(self,unlabel_loader, test_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer2 = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        best_res = {"f1":-1}
        best_state, best_test_scores = None, None
        lr=self.learning_rate

        pre_loss, worse_count = float("inf"), 0
        early_stopping_ct = 0
        for epoch in range(0, self.epoches_1):
            
            self.current_epoch = epoch
            batch_cnt, epoch_loss = 0, 0.0
            batch_cnt2, epoch_loss2 = 0, 0.0
            epoch_time_start = time.time()
            
            kpi_inputs = []
            log_inputs = []
            kpi_outputs = []
            log_outputs = []
            labels = []
            
            # train generator
            self.model.train()
            self.discriminator.eval()
            self.loss_fusion.eval()
            for batch_input in unlabel_loader:
                kpi_inputs.append(batch_input["kpi_features"])
                log_inputs.append(batch_input["log_features"])
                labels.append(batch_input["labels"])
                optimizer.zero_grad()
                res = self.model(self.__input2device(batch_input), True) if self.open_unmatch_zoomout else self.model(self.__input2device(batch_input))
                # macs1, params1 = profile(self.model, inputs=(self.__input2device(batch_input),))



                if self.data_type == "fuse":
                    # loss = res["loss"][0] + res["loss"][1]
                    # loss = res["fusion_loss"]
                    loss = res["loss"]
                    kpi_outputs.append(res["output"][1].detach().cpu().numpy())
                    log_outputs.append(res["output"][0].detach().cpu().numpy())
                elif self.data_type == "kpi":
                    kpi_outputs.append(res["output"].detach().cpu().numpy())
                    log_outputs.append(log_inputs[-1])
                    loss = res["loss"]
                elif self.data_type == "log":
                    kpi_outputs.append(kpi_inputs[-1])
                    log_outputs.append(res["output"].detach().cpu().numpy())
                    loss = res["loss"]
                loss = loss.mean()
                if self.open_gan:
                    if self.data_type == "fuse":
                        if self.open_gan_sep:
                            # macs2, params2 = profile(self.discriminator, inputs=((self.__input2device(batch_input),res,True),))
                            # print(" the macs is {}G and params is {}M".format(str((macs1) / (1000 ** 3)),
                            #                                                   str((params1) / (1000 ** 2))))
                            loss += self.discriminator.get_loss_sep(self.__input2device(batch_input),res,flag=True)["deceive_loss"] if self.open_unmatch_zoomout else self.discriminator.get_loss_sep(self.__input2device(batch_input),res,flag=False)["deceive_loss"]
                        else:
                            loss += self.discriminator.get_loss(self.__input2device(batch_input),res,flag=True)["deceive_loss"] if self.open_unmatch_zoomout else self.discriminator.get_loss(self.__input2device(batch_input),res,flag=False)["deceive_loss"]
                            # loss += self.discriminator.get_loss(self.__input2device(batch_input),res,flag=True)["deceive_loss"] 
                    else:
                        loss += self.discriminator.get_loss(self.__input2device(batch_input),res,flag=False)["deceive_loss"]
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_loss = epoch_loss / batch_cnt
            logging.info("Epoch {}/{}, train generator loss: {:.5f}".format(epoch, self.epoches_1, epoch_loss))

            kpi_inputs = np.concatenate(kpi_inputs,axis=0).reshape(-1,self.kpi_c)
            kpi_outputs = np.concatenate(kpi_outputs,axis=0).reshape(-1,self.kpi_c)
            log_inputs = np.concatenate(log_inputs,axis=0).reshape(-1,self.log_c)
            log_outputs = np.concatenate(log_outputs,axis=0).reshape(-1,self.log_c)
            labels = np.concatenate(labels,axis=0).reshape(-1)
            # plot_labeled_2curve(kpi_outputs,kpi_inputs,labels,"none",f"train_kpi_pred_gt_label_{self.current_epoch}")
            # plot_labeled_2curve(log_outputs,log_inputs,"none","none",f"train_log_pred_gt_label_{self.current_epoch}")

            # train discriminator
            if self.open_gan:
                self.model.eval()
                self.discriminator.train()
                self.loss_fusion.eval()
                for batch_input in unlabel_loader:
                    optimizer2.zero_grad()
                    res = self.model(self.__input2device(batch_input), flag=True)
                    loss = self.discriminator.get_loss(self.__input2device(batch_input),res)["loss"]
                    loss = loss.mean()
                    loss.backward()
                    optimizer2.step()
                    epoch_loss2 += loss.item()
                    batch_cnt2 += 1
                epoch_loss2 = epoch_loss2 / batch_cnt2
                logging.info("Epoch {}/{}, train discriminator loss: {:.5f}".format(epoch, self.epoches_1, epoch_loss2))
               
            epoch_time_elapsed = time.time() - epoch_time_start
            self.train_time.append(epoch_time_elapsed)
            start_time = time.time()
            if self.evaluation_sep:
                test_results = self.evaluate_sep(test_loader, datatype="Test")
            else:
                test_results,test_embeds = self.evaluate(test_loader, datatype="Test")
            logging.info("*** Test time is {}".format(time.time() - start_time))
            logging.info("Epoch {}/{}, test f1: {:.5f} rc: {:.5f} pc: {:.5f}. [{:.2f}s]".format(epoch, self.epoches_1, test_results["f1"],test_results["rc"],test_results["pc"],epoch_time_elapsed))

            if test_results["f1"] > best_res["f1"]:
                best_res = test_results
                # best_embeds = test_embeds
                # np.save(f"temp/embeds_{self.kwadrgs['dataset']}_{self.kwargs['open_gan']}_{self.kwargs['open_unmatch_zoomout']}_{self.kwargs['unmatch_k']}.npy",test_embeds["embeds"])
                # np.save(f"temp/distance_{self.kwargs['dataset']}_{self.kwargs['open_gan']}_{self.kwargs['open_unmatch_zoomout']}_{self.kwargs['unmatch_k']}.npy",test_embeds["distance"])
                # np.save(f"temp/labels_{self.kwargs['dataset']}.npy",test_embeds["labels"])

                best_test_scores = test_results
                best_state = copy.deepcopy(self.model.state_dict())
                early_stopping_ct = 0
            else:
                early_stopping_ct +=1
                # lr = adjust_learning_rate(optimizer, lr)
                logging.info(f'early stopping epoch {early_stopping_ct}/{self.patience}')

            if early_stopping_ct >= self.patience :
                logging.info("Early stopping")
                break

        self.save_model(best_state)
        self.load_model(self.model_save_file)
        

        test_results,test_embeds = self.evaluate(test_loader, datatype="Test")
       
        logging.info("*** Test F1 {:.4f}  of unsupervised traning".format(test_results["f1"]))
        logging.info(f"*** Best F1 {best_test_scores}  of unsupervised traning")


        return best_test_scores
    

    