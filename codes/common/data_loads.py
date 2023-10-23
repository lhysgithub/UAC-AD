import os
import pickle
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from common.semantics import FeatureExtractor
import logging
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, RobustScaler
# from src.pc import pc
# from src.utils import get_causal_chains, plot
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def load_sessions(data_dir,**keywds): #both log and kpi
    logging.info("Load from {}".format(data_dir))
    with open(os.path.join(data_dir, "train.pkl"), "rb") as fr:
        train = pickle.load(fr)
    if keywds["dataset"] == "yzh": 
        unlabel = {}
    else:
        with open(os.path.join(data_dir, "unlabel.pkl"), "rb") as fr:
            unlabel = pickle.load(fr)
    with open(os.path.join(data_dir, "test.pkl"), "rb") as fr:
        test = pickle.load(fr)
    return train, unlabel, test

class myDataset(Dataset):
    def __init__(self, sessions,window_size=100, test_flag=False):
        self.data = []
        self.window=[]
        self.idx2id = {}
        self.test_flag = test_flag
        for idx, block_id in enumerate(sessions.keys()):
            self.idx2id[idx] = block_id
            item = sessions[block_id]
            sample = {
                'idx': idx,
                'label': int(item['label']),
                'kpi_features': item['kpis'],
                'unmatched_kpi_features': item['unmatched_kpi_features'],
                # 'log_sequence':item['seqs'],
                'raw_logs':item["logs"],
                'log_features': item['log_features'],
                
                # 'kpi_cluster':item["cluster_kpis"],
                # "log_cluster":item["cluster_log"]
            }
            self.data.append(sample)
        
        if test_flag:
            for i in range(window_size,len(self.data),window_size):
                self.window.append(self.data[i-window_size:i])
        else:
            for i in range(len(self.data)-window_size):
                self.window.append(self.data[i:i+window_size])

        # self.window=[]
        # for idx in range(len(self.data)-window_size):
        #     self.window.append(self.data[idx:idx+window_size])


    def __len__(self):
        return len(self.window)
    def __getitem__(self, idx):
        # log_features=np.array([x['log_features'] for x in self.window[idx]][:-1])
        # kpi_features=np.array([x['kpis'] for x in self.window[idx]][:-1])
        # label=self.window[idx][-1]["label"]
        # metric_label=self.window[idx][-1]["kpi_cluster"]
        # log_label=self.window[idx][-1]["log_cluster"]
        # label=self.data[idx]["label"]
        # log_features=self.data[idx]["log_features"]
        log_template=[]
        kpis = []
        labels=[]
        kpis2 = []
        # log2 = []
        for block in self.window[idx]:
            kpis.append(block["kpi_features"]) # todo 待聚合
            log_template.append(block["log_features"])
            labels.append(block["label"])
            # new_id = np.random.randint(len(self.data))
            # while (self.data[new_id]["log_features"] == block["log_features"]).all() or np.abs(self.data[new_id]["kpi_features"]-block["kpi_features"]).mean()<0.15:
            #     new_id = np.random.randint(len(self.data))
            # new_kpi = self.data[new_id]["kpi_features"]
            # old_kpi = block["kpi_features"]
            # kpi_dis = np.abs(self.data[new_id]["kpi_features"]-block["kpi_features"]).mean()
            # new_log = self.data[new_id]["log_features"]
            # old_log = block["log_features"]
            # log_dis = np.abs(self.data[new_id]["log_features"]-block["log_features"]).mean()
            kpis2.append(block["unmatched_kpi_features"])
        
        # new_idx = np.random.randint(len(self.window))
        # # new_idx = idx
        # batch_length = len(self.window[new_idx])
        # for block in self.window[new_idx]:
        #     noise = np.random.randn(len(block["kpi_features"]))
        #     new_kpi = block["kpi_features"] + noise
        #     kpis2.append(new_kpi) # todo 待聚合
        #     # noise = np.random.randn(len(block["log_features"]))
        #     # new_log = block["log_features"] + noise
        #     # log2.append(new_log)
            
            
        return {
                # "kpi_features":torch.FloatTensor(np.array(kpis).mean(axis=1)),
                "kpi_features":torch.FloatTensor(np.array(kpis)),
                "log_features":torch.FloatTensor(np.array(log_template)),
                "labels":torch.tensor(labels),
                "unmatched_kpi_features":torch.FloatTensor(np.array(kpis2)),
                # "unmatched_log_features":torch.FloatTensor(np.array(log2)),
            }
            
        # return torch.FloatTensor(np.array(log_template)),torch.tensor(labels)
    def __get_session_id__(self, idx):
        return self.idx2id[idx]
    
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

def get_features(chunks):
    kpis = []
    labels = []
    logs = []
    for k,v in chunks.items():
        kpis.append(v["kpis"])
        labels.append(v["label"])
        logs.append(v["log_features"])
    kpis = np.array(kpis)
    labels = np.array(labels)
    logs = np.array(logs)
    return {"kpis":kpis,"labels":labels,"logs":logs}

def kpi_selection(train_kpis,unlabel_kpi,test_kpi,**params):
    stds = []
    for i in range(train_kpis.shape[1]):
        stds.append(np.std(train_kpis[:,i]))
    stds = np.array(stds)
    threshold= np.percentile(stds,params["kpi_ratio"])
    pred=[1 if d < threshold  else 0 for d in stds]
    if params["kpi_with_high_std"]:
        pred=[1 if d > threshold  else 0 for d in stds]
    train_kpis_ = []
    unlabel_kpi_ = []
    test_kpi_ = []
    for i in range(len(pred)):
        if pred[i]:
            train_kpis_.append(train_kpis[:,i])
            if len(unlabel_kpi)>0:
                unlabel_kpi_.append(unlabel_kpi[:,i])
            test_kpi_.append(test_kpi[:,i])
    train_kpis = np.array(train_kpis_).T
    if len(unlabel_kpi)>0:
        unlabel_kpi = np.array(unlabel_kpi_).T
    test_kpi = np.array(test_kpi_).T
    # if not os.path.exists("../data/cliped/train.npy"):
        #     np.save("../data/cliped/train.npy",train_kpis)
        #     np.save("../data/cliped/test.npy",test_kpi)
        #     np.save("../data/cliped/test_label.npy",test_labels)
    return train_kpis,unlabel_kpi,test_kpi
    

def reconstruction_chunks(features,chunks):
    cnt = 0
    new_chunks = {}
    for k,v in chunks.items():
        v["kpis"] = features["kpis"][cnt]
        v["log_features"] = features["logs"][cnt]
        new_chunks[k] = v
        cnt += 1
    return new_chunks
    

def normalization(train_chunks, unlabel_chunks, test_chunks,**params):
    train_features = get_features(train_chunks)
    unlabel_features = get_features(unlabel_chunks)
    test_features = get_features(test_chunks)
    
    if params["dataset"] == "original":
        train_features["kpis"] = np.mean(train_features["kpis"],axis=-2)
        unlabel_features["kpis"] = np.mean(unlabel_features["kpis"],axis=-2)
        test_features["kpis"] = np.mean(test_features["kpis"],axis=-2)
        
    
    # plt.figure(figsize=(8*2,6*2))
    # plot_labeled_curve(unlabel_kpi,train_labels,"train_kpis")
    # plot_labeled_curve(train_kpis,train_labels,"train_kpis")
    # plot_labeled_curve(test_kpi,test_labels,"test_kpi")
    
    if params["open_kpi_normalization"]:
        train_features["kpis"], scaler = normalize_data(train_features["kpis"], scaler=None) # 此处假设训练集的分布与测试集分布相似，但是目前C的数据并不符合这个假设
        unlabel_features["kpis"], _ = normalize_data(unlabel_features["kpis"], scaler=scaler)
        test_features["kpis"], _ = normalize_data(test_features["kpis"], scaler=scaler)
    
    if params["open_log_normalization"]:
        train_features["logs"], scaler = normalize_data(train_features["logs"], scaler=None)
        unlabel_features["logs"], _ = normalize_data(unlabel_features["logs"], scaler=scaler)
        test_features["logs"], _ = normalize_data(test_features["logs"], scaler=scaler)
    
    # params["scaler"] = scaler
    # plot_curve(train_kpis,"train_kpis_norm")
    # plot_curve(test_kpi,"test_kpi_norm")
    
    # 标准差过滤
    if params["open_kpi_select"]:
        train_features["kpis"],unlabel_features["kpis"],test_features["kpis"] = kpi_selection(train_features["kpis"],unlabel_features["kpis"],test_features["kpis"],**params)
 
    new_train_chunks = reconstruction_chunks(train_features,train_chunks)
    new_unlabel_chunks = reconstruction_chunks(unlabel_features,unlabel_chunks)
    new_test_chunks = reconstruction_chunks(test_features,test_chunks)
   
    return new_train_chunks,new_unlabel_chunks,new_test_chunks

def construct_unmatched_data(data, params):
    temp_data = []
    for idx,dic in enumerate(data.items()) :
        temp_data.append(dic)
    
    for idx,dic in enumerate(temp_data) :
        old_kpis = temp_data[idx][1]["kpis"]
        old_log = temp_data[idx][1]["log_features"]
        new_id = np.random.randint(len(temp_data))
        while (temp_data[new_id][1]["log_features"] == old_log).all() or np.abs(temp_data[new_id][1]["kpis"]-old_kpis).mean()<params["theta"]:
            new_id = np.random.randint(len(temp_data))
        new_kpis = temp_data[new_id][1]["kpis"]
        temp_data[idx][1]["unmatched_kpi_features"] = new_kpis
    idx = 0
    for k,v in data.items() :
        data[k]["unmatched_kpi_features"] = temp_data[idx][1]["unmatched_kpi_features"]
        idx+=1
    return data

class Process():
    def __init__(self, var_nums, labeled_train, unlabel_train, unsupervised_train,test_chunks, supervised=False, **kwargs):
        self.var_nums = var_nums
        self.kpi_kns=None
        self.log_kns=None
        self.ext = FeatureExtractor(**kwargs)
        self.__train_ext(labeled_train, unlabel_train)
        
        del labeled_train
        # unlabel_train = unsupervised_train
        # labeled_train = self.ext.transform(labeled_train)
        
        if not supervised:
            unlabel_train = self.ext.transform(unsupervised_train, datatype="unlabel train")
            # unlabel_train = self.__transform_kpi(unlabel_train)
            # unlabel_train=self.__transform_cluster(unlabel_train)

        test_chunks = self.ext.transform(test_chunks, datatype="test")
        # # labeled_train = self.__transform_kpi(labeled_train)
        # test_chunks = self.__transform_cluster(test_chunks)
        
        labeled_train, unlabel_train, test_chunks = normalization(unlabel_train, unlabel_train, test_chunks,**kwargs)

        logging.info('Data loaded done!')
        
        self.unlabel_train = construct_unmatched_data(unlabel_train, kwargs)
        self.test_chunks = construct_unmatched_data(test_chunks, kwargs)
        
        self.dataset = {
            # 'train': myDataset(labeled_train),
            'unlabel': myDataset(self.unlabel_train,window_size=kwargs["window_size"]) if not supervised else None,
            'test':  myDataset(self.test_chunks,window_size=kwargs["window_size"],test_flag=True),
        }
        
    def __train_ext(self, a, b):
        a.update(b)
        self.ext.fit(a)
    
    def __transform_kpi(self, chunks):
        for id, dict in chunks.items():
            kpis = dict['kpis']

            if kpis.shape[0] != sum(self.var_nums): kpis = kpis.T
            chunks[id]['kpi_features'] = []
            pre_num = 0
            for num in self.var_nums:
                chunks[id]['kpi_features'].append(kpis[pre_num:pre_num+num, :])
                pre_num += num
            # kpis = kpis.mean(1) #这一句是专门求10s平均，为了跟日志数据对齐
            # logs = dict["log_features"]
            # chunk_kpi = np.concatenate((kpis, logs))
            # chunks[id]["features"]=chunk_kpi
        return chunks

    def __transform_cluster(self,chunks):
        total_kpi=[]
        total_log=[]
        for id,dict in tqdm(chunks.items()):
            total_kpi.append(chunks[id]["kpis"])
            total_log.append(chunks[id]["log_features"])
        size = len(chunks)
        #分别做聚类，得到每个窗口的聚类标签，方便后面做模态间和模态内的无监督学习
        if not self.kpi_kns and not self.log_kns:
            self.kpi_kns = AgglomerativeClustering(linkage="ward", n_clusters=10)
            self.log_kns = AgglomerativeClustering(linkage="ward", n_clusters=10)
            predict_kpi = self.kpi_kns.fit_predict(np.array(total_kpi).reshape(size, -1))
            predict_log = self.log_kns.fit_predict(np.array(total_log).reshape(size, -1))

        else:
            predict_kpi = self.kpi_kns.predict(np.array(total_kpi).reshape(size, -1))
            predict_log = self.log_kns.predict(np.array(total_log).reshape(size, -1))

            chunks[id]["cluster_kpi"]=predict_kpi[id]
            chunks[id]["cluster_log"]=predict_log[id]
        return chunks

    def __transform_graph(self,chunks):
        pass
        #
        #
        # # columns_name=["user_cpu","system_cpu","io_wait","idle","rkB_s","wkB_s","util","memused","commit","rxkB_s","txkB_s"]+[ "log_{}".format(k) for k in range(self.ext.cluster_y_pred.max()+1)]
        # columns_name=["user_cpu","system_cpu","io_wait","idle","rkB_s","wkB_s","util","memused","commit","rxkB_s","txkB_s"]+[ k for k in self.ext.log2id_train.keys()]
        # columns_name.remove("padding")
        #
        # df=pd.DataFrame(data=total_kpi,columns=columns_name)
        #
        # unseen_tem=df.loc[:, (df == 0).all()].columns
        # df.drop(columns=unseen_tem,inplace=True)
        # df.reset_index(inplace=True)
        # labels = df.columns.to_list()
        # p = pc(
        #     suff_stat={"C": df.corr().values, "n": df.shape[0]},
        #     verbose=True
        # )
        #
        # # DFS 因果关系链
        # for num in range(len(labels)):
        #     print(get_causal_chains(p, start=num, labels=labels))
        #
        # # 画图
        # plot(p, labels, "./causal_graph",)
        # with open("./graph_adj","wb") as f:
        #     pickle.dump(p,f)
        # df.to_csv("df.csv",index=False)
        # print("save pc and figure")
        # return chunks,p
