import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

import argparse
parser = argparse.ArgumentParser()
### Model params
parser.add_argument("--supervised", action="store_true")
parser.add_argument("--gpu", default=True, type=lambda x: x.lower() == "true")
parser.add_argument("--epoches", default=[50, 50], type=int, nargs='+')
parser.add_argument("--batch_size", default=128, type=int) # 可调
parser.add_argument("--confidence", default=0.92, type=float)
parser.add_argument("--alpha", default=0.5, type=float)
parser.add_argument("--learning_rate", default=0.001, type=float) # 可调 done
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--random_seed", default=42, type=int) # 可调
parser.add_argument("--optim", default=-1, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--kpi_ratio", default=40, type=int)
parser.add_argument("--k", default=1, type=int)
parser.add_argument("--kpi_with_high_std", default=False, type=str2bool)
# parser.add_argument("--open_attention_discrepancy", default=False, type=str2bool)
parser.add_argument("--gpu_device", default="0", type=str)
parser.add_argument("--open_kpi_select", default=False, type=str2bool)
parser.add_argument("--open_min_max", default=False, type=str2bool)
parser.add_argument("--open_position_embedding", default=False, type=str2bool)
parser.add_argument("--sigma_matrix", default=False, type=str2bool)
parser.add_argument("--feature_type", default="template_appear", type=str, choices=["word2vec", "sequential","template_count","template_appear"])
parser.add_argument("--data", type=str, default="../data/chunk_10")
parser.add_argument("--dataset", type=str, default="original")
# parser.add_argument("--dataset", type=str, default="yzh")
# parser.add_argument("--data", type=str, default="../data/data3")
# parser.add_argument("--dataset", type=str, default="zte")
# parser.add_argument("--data", type=str, default="../data/zte2")
parser.add_argument("--open_kpi_normalization", default=True, type=str2bool)
parser.add_argument("--open_log_normalization", default=False, type=str2bool)
parser.add_argument("--open_narrowing_modal_gap", default=False, type=str2bool) # True 大规模测试时
parser.add_argument("--open_feature2", default=False, type=str2bool)
parser.add_argument("--open_expand_anomaly_gap", default=False, type=str2bool) # True 大规模测试时
parser.add_argument("--evaluation_sep", default=False, type=str2bool)
parser.add_argument("--open_gan_sep", default=False, type=str2bool)
parser.add_argument("--open_gan", default=True, type=str2bool)
parser.add_argument("--open_unmatch_zoomout", default=True, type=str2bool)
parser.add_argument("--unmatch_k", default=1, type=int)
parser.add_argument("--run_times", default=0, type=int)
parser.add_argument("--criterion", default="l1", type=str, choices=["l1", "mse"])

##### Munual params
parser.add_argument("--window_size", default=30, type=int)
parser.add_argument("--hidden_size", default=64, type=int, help="Dim of the commnon feature space") # 可调
# Fuse params
parser.add_argument("--data_type", default="fuse", choices=["fuse", "log", "kpi"])
parser.add_argument("--fuse_type", default="multi_modal_self_attn", choices=["concat", "cross_attn", "sep_attn","multi_modal_self_attn"])
parser.add_argument("--attn_type", default="add", choices=["dot", "add","qkv"])

### Kpi params
parser.add_argument("--inner_dropout", default=0.5, type=float)

### Log params
parser.add_argument("--log_layer_num", default=4, type=int)
parser.add_argument("--log_dropout", default=0.1, type=float)
parser.add_argument("--transformer_hidden", default=1024, type=int)

# Word params
parser.add_argument("--word2vec_model_type", default="fasttext", type=str, choices=["naive","fasttext","skip-gram"])
parser.add_argument("--word_embedding_dim", default=32, type=int)
parser.add_argument("--word_window", default=5, type=int)
parser.add_argument("--word2vec_epoch", default=50, type=int)

### Control params
parser.add_argument("--pre_model", default=None, type=str)
parser.add_argument("--word2vec_save_dir", default="../trained_wv/", type=str)
parser.add_argument("--result_dir", default="../result21/", type=str)
parser.add_argument("--main_model", default="hades", choices=["hades", "join-hades", "concat-hades", "sep-hades", "agn-hades", "one-hades", "met-hades", "anno-hades"])

params = vars(parser.parse_args())

def get_res(filename):
    with open(filename,"r") as f:
        lines = f.readlines() 
        line = lines[2]
        f1 = float(line.split(":")[1].split("\t")[0]) 
        rc = float(line.split(":")[2].split("\t")[0])
        pc = float(line.split(":")[3])
    return {"f1":f1, "rc":rc, "pc":pc}

def get_one_data(params):
    hash_id = f'{params["dataset"]}_{params["window_size"]}_{params["k"]}_{params["criterion"]}_{params["open_kpi_normalization"]}_{params["open_log_normalization"]}_{params["feature_type"]}_{params["open_min_max"]}_{params["open_kpi_select"]}_{params["kpi_ratio"]}_{params["kpi_with_high_std"]}_{params["attn_type"]}_{params["fuse_type"]}_{params["data_type"]}_{params["hidden_size"]}_{params["open_position_embedding"]}_{params["sigma_matrix"]}_{params["open_narrowing_modal_gap"]}_{params["open_feature2"]}_{params["open_expand_anomaly_gap"]}_{params["evaluation_sep"]}_{params["open_gan"]}_{params["open_unmatch_zoomout"]}_{params["open_gan_sep"]}_{params["unmatch_k"]}_{params["run_times"]}'
    # hash_id = f'{params["dataset"]}_{params["window_size"]}_{params["k"]}_{params["criterion"]}_{params["open_kpi_normalization"]}_{params["open_log_normalization"]}_{params["feature_type"]}_{params["open_min_max"]}_{params["open_kpi_select"]}_{params["kpi_ratio"]}_{params["kpi_with_high_std"]}_{params["attn_type"]}_{params["fuse_type"]}_{params["data_type"]}_{params["hidden_size"]}_
    # {params["open_position_embedding"]}_{params["sigma_matrix"]}_{params["open_narrowing_modal_gap"]}_{params["open_feature2"]}_{params["open_expand_anomaly_gap"]}_{params["evaluation_sep"]}_{params["open_gan"]}_{params["open_unmatch_zoomout"]}_{params["open_gan_sep"]}_{params["unmatch_k"]}_{params["run_times"]}'
    # original_30_1_l1_True_False_template_appear_False_False_40_False_add_multi_modal_self_attn_fuse_64_
    # False_False_True_False_True_False_True_True_False_-6_3
    # original_30_1_l1_True_False_template_appear_False_False_40_False_add_multi_modal_self_attn_fuse_64_False_False_True_False_True_False_True_True_False_-6_3
    filename = os.path.join(params["result_dir"], hash_id) + f"/info_score.txt"
    return get_res(filename)

def get_all_data():
    temp0 = []
    for k in range(2):
        if k == 1:
            params["kpi_with_high_std"] = True
        else:
            params["kpi_with_high_std"] = False
        temp1 = [] 
        for j in range(3,8):
            temp2 = []
            params["kpi_ratio"] = j*10
            for i in range(5):
                temp3 = []
                params["run_times"] = i
                res = get_one_data(params)
                temp3.append(res["f1"])
                temp3.append(res["rc"])
                temp3.append(res["pc"])
                temp2.append(temp3)
            temp1.append(temp2)
        temp0.append(temp1)
    return np.array(temp0)

def get_all_data_for_unmatched_k():
    params["window_size"] = 100
    params["open_narrowing_modal_gap"] = True
    params["open_expand_anomaly_gap"] = True
    temp1 = [] 
    for j in range(2,21,2):
        temp2 = []
        params["unmatch_k"] = j
        for i in range(5):
            temp3 = []
            params["run_times"] = i
            res = get_one_data(params)
            temp3.append(res["f1"])
            temp3.append(res["rc"])
            temp3.append(res["pc"])
            temp2.append(temp3)
        temp1.append(temp2)
    return np.array(temp1)

def get_all_data_for_unmatched_k_v2():
    # params["window_size"] = 30
    # params["open_narrowing_modal_gap"] = True
    # params["open_expand_anomaly_gap"] = True
    temp1 = [] 
    # 1 3 5 7 9 11 13 15 17 19 21
    for j in range(2,21,2):
    # for j in range(1,21):
    # for j in range(1,18):
        temp2 = []
        params["unmatch_k"] = j
        for i in range(5):
            temp3 = []
            params["run_times"] = i
            res = get_one_data(params)
            temp3.append(res["f1"])
            temp3.append(res["rc"])
            temp3.append(res["pc"])
            temp2.append(temp3)
        temp1.append(temp2)
    return np.array(temp1)

def get_all_data_for_hidden_size():
    temp1 = [] 
    for j in [16,32,64,128,256,512]:
    # for j in [16,32,64,128,256]:
        temp2 = []
        params["hidden_size"] = j
        for i in range(5):
            temp3 = []
            params["run_times"] = i
            res = get_one_data(params)
            temp3.append(res["f1"])
            temp3.append(res["rc"])
            temp3.append(res["pc"])
            temp2.append(temp3)
        temp1.append(temp2)
    return np.array(temp1)

def get_all_data_for_window_size():
    temp1 = [] 
    for j in [10,20,30,40,50,60,70,80,90,100]:
        temp2 = []
        params["window_size"] = j
        for i in range(5):
            temp3 = []
            params["run_times"] = i
            res = get_one_data(params)
            temp3.append(res["f1"])
            temp3.append(res["rc"])
            temp3.append(res["pc"])
            temp2.append(temp3)
        temp1.append(temp2)
    return np.array(temp1)

def plot_hidden_size():
    params["unmatch_k"] = 16
    params["dataset"] = "yzh"
    params["data"] = "../data/data3"
    res2 = get_all_data_for_hidden_size()
    params["dataset"] = "zte"
    params["data"] = "../data/zte2"
    res1 = get_all_data_for_hidden_size()
    params["dataset"] = "original"
    params["data"] = "../data/chunk_10"
    params["open_narrowing_modal_gap"] = True
    params["open_expand_anomaly_gap"] = True
    res0 = get_all_data_for_hidden_size()
    plt.cla()
    res = np.mean(res0,axis=-2) # median # mean # max
    res2 = np.mean(res2,axis=-2)
    res1 = np.mean(res1,axis=-2)
    res3 = np.array([res,res2,res1]).mean(axis=0)
    print(res3[:,0])
    x = [16,32,64,128,256,512]
    # x = [16,32,64,128,256]
    plt.plot(x,res[:,0],"p-c",label="Dataset A",alpha=0.7)
    plt.plot(x,res2[:,0],"^:g",label="Dataset B",alpha=0.7)
    plt.plot(x,res1[:,0],"P--b",label="Dataset C",alpha=0.7)
    print(res3[:,0])
    plt.plot(x,res3[:,0],"s-.r",label="Average",alpha=0.7)
    plt.grid()
    plt.xlabel("hidden_size")
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    # plt.legend()
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    plt.savefig(f"temp/hidden_size.pdf")

def plot_window_size():
    params["unmatch_k"] = 16
    params["dataset"] = "yzh"
    params["data"] = "../data/data3"
    res2 = get_all_data_for_window_size()
    params["dataset"] = "zte"
    params["data"] = "../data/zte2"
    res1 = get_all_data_for_window_size()
    params["dataset"] = "original"
    params["data"] = "../data/chunk_10"
    params["open_narrowing_modal_gap"] = True
    params["open_expand_anomaly_gap"] = True
    res0 = get_all_data_for_window_size()
    plt.cla()
    res = np.mean(res0,axis=-2) # mean median max
    res2 = np.mean(res2,axis=-2)
    res1 = np.mean(res1,axis=-2)
    res3 = np.array([res,res2,res1]).mean(axis=0)
    print(res3[:,0])
    x = [10,20,30,40,50,60,70,80,90,100]
    plt.plot(x,res[:,0],"p-c",label="Dataset A",alpha=0.7)
    plt.plot(x,res2[:,0],"^:g",label="Dataset B",alpha=0.7)
    plt.plot(x,res1[:,0],"P--b",label="Dataset C",alpha=0.7)
    print(res3[:,0])
    plt.plot(x,res3[:,0],"s-.r",label="Average",alpha=0.7)
    plt.grid()
    plt.xlabel("window_size")
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    # plt.legend()
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    plt.savefig(f"temp/window_size.pdf")

def plot_unmatched_k():
    res0 = get_all_data_for_unmatched_k()
    params["dataset"] = "yzh"
    params["data"] = "../data/data3"
    res2 = get_all_data_for_unmatched_k()
    plt.cla()
    res = np.median(res0,axis=-2)
    res2 = np.median(res2,axis=-2)
    res3 = np.array([res,res2]).mean(axis=0)
    x = [(i+1)*0.2+1 for i in range(len(res))]
    # plt.plot(x,res[:,0],label="Dataset A")
    # plt.plot(x,res2[:,0],label="Dataset B")
    print(res3[:,0])
    plt.plot(x,res3[:,0],label="k")
    plt.grid()
    plt.xlabel("k")
    plt.ylabel("F1")
    plt.ylim(0.85,0.95)
    plt.legend()
    
    plt.savefig(f"temp/k.pdf")
    
def plot_unmatched_k_v2():
    params["window_size"] = 40
    params["dataset"] = "yzh"
    params["data"] = "../data/data3"
    res2 = get_all_data_for_unmatched_k_v2()
    params["dataset"] = "zte"
    params["data"] = "../data/zte2"
    res1 = get_all_data_for_unmatched_k_v2()
    params["dataset"] = "original"
    params["data"] = "../data/chunk_10"
    params["open_narrowing_modal_gap"] = True
    params["open_expand_anomaly_gap"] = True
    res0 = get_all_data_for_unmatched_k_v2()
    plt.cla()
    res = np.mean(res0,axis=-2) # mean median max
    res1 = np.mean(res1,axis=-2) 
    res2 = np.mean(res2,axis=-2)
    res3 = np.array([res,res1,res2]).mean(axis=0)
    x = [(i+1)*0.02 for i in range(len(res))]
    plt.plot(x,res[:,0],"p-c",label="Dataset A",alpha=0.7)
    plt.plot(x,res2[:,0],"^:g",label="Dataset B",alpha=0.7)
    plt.plot(x,res1[:,0],"P--b",label="Dataset C",alpha=0.7)
    print(res3[:,0])
    plt.plot(x,res3[:,0],"s-.r",label="Average",alpha=0.7)
    plt.grid()
    plt.xlabel("alpha")
    plt.ylabel("F1-score")
    # plt.ylim(0.85,0.95)
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.07),ncol=4)
    
    plt.savefig(f"temp/k.pdf")

def plot_kpi_ratio_and_kpi_with_high_std():
    res0 = get_all_data()
    plt.cla()
    res = np.mean(res0,axis=-2)
    std = np.std(res0,axis=-2)
    min_ = np.min(res0,axis=-2)
    max_ = np.max(res0,axis=-2)
    x = [i +3 for i in range(len(res[0]))]
    plt.plot(x,res[0,:,0],"k",label="low_std")
    plt.plot(x,res[1,:,0],'b',label="high_std")
    plt.legend()
    plt.savefig("temp/kpi_ratio_and_kpi_with_high_std.pdf")
    
    plt.cla()
    plt.plot(x,std[0,:,0],"k-.",label="low_std_s")
    plt.plot(x,std[1,:,0],"b-.",label="high_std_s")
    plt.legend()
    plt.savefig("temp/kpi_ratio_and_kpi_with_high_std_s.pdf")
    
    plt.cla()
    plt.plot(x,min_[0,:,0],"k-.",label="low_std_min")
    plt.plot(x,min_[1,:,0],"b-.",label="high_std_min")
    plt.legend()
    plt.savefig("temp/kpi_ratio_and_kpi_with_high_std_min.pdf")
    
    plt.cla()
    plt.plot(x,max_[0,:,0],"k-.",label="low_std_max")
    plt.plot(x,max_[1,:,0],"b-.",label="high_std_max")
    plt.legend()
    plt.savefig("temp/kpi_ratio_and_kpi_with_high_std_max.pdf")

def caculate_mean_std(name):
    temp = []
    for i in range(5):
        filename = name + f"_{i}/info_score.txt"
        res = get_res(filename)
        temp1 = []
        temp1.append(res["f1"])
        temp1.append(res["rc"])
        temp1.append(res["pc"])
        temp.append(temp1)
    temp = np.array(temp)
    print(f"f1:{temp[:,0].mean()} rc:{temp[:,1].mean()} pr:{temp[:,2].mean()}")

def caculate_mean_std_k(name,k):
    temp = []
    # ks = [0,2,4]
    # for i in ks:
    for i in range(k,k+5):
        filename = name + f"_{i}/info_score.txt"
        res = get_res(filename)
        temp1 = []
        temp1.append(res["f1"])
        temp1.append(res["rc"])
        temp1.append(res["pc"])
        temp.append(temp1)
    temp = np.array(temp)
    print(temp[:,0])
    print(f"f1:{temp[:,0].mean()} rc:{temp[:,1].mean()} pr:{temp[:,2].mean()}")
    print(f"f1:{temp[:,0].std()} rc:{temp[:,1].std()} pr:{temp[:,2].std()}")
    index = np.argmax(temp[:,0]) 
    # f1 = temp[:,0].mean()
    # rc = temp[:,1].mean()
    # pc = temp[:,2].mean()
    f1 = temp[index,0]
    rc = temp[index,1]
    pc = temp[index,2]
    return {"f1":f1,"rc":rc,"pc":pc}

def display_ablation():
    
    if params["dataset"] == "original":
        params["open_narrowing_modal_gap"] = True
        params["open_expand_anomaly_gap"] = True
    
    data_type = []
    fuse_type = []
    open_gan = []
    open_unmatch = []
    unmatch_k = []
    # # L         G
    # data_type.append("log")
    # fuse_type.append("multi_modal_self_attn")
    # open_gan.append(False)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # M         G
    # data_type.append("kpi")
    # fuse_type.append("multi_modal_self_attn")
    # open_gan.append(False)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # L         G,D
    # data_type.append("log")
    # fuse_type.append("multi_modal_self_attn")
    # open_gan.append(True)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # M         G,D
    # data_type.append("kpi")
    # fuse_type.append("multi_modal_self_attn")
    # open_gan.append(True)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # M,L	CA	G
    # data_type.append("fuse")
    # fuse_type.append("cross_attn")
    # open_gan.append(False)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # M,L	SA	G
    # data_type.append("fuse")
    # fuse_type.append("sep_attn")
    # open_gan.append(False)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # M,L	C	G
    # data_type.append("fuse")
    # fuse_type.append("concat")
    # open_gan.append(False)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # M,L	MSA	G
    # data_type.append("fuse")
    # fuse_type.append("multi_modal_self_attn")
    # open_gan.append(False)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # M,L	CA	G,D
    # data_type.append("fuse")
    # fuse_type.append("cross_attn")
    # open_gan.append(True)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # M,L	SA	G,D
    # data_type.append("fuse")
    # fuse_type.append("sep_attn")
    # open_gan.append(True)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # M,L	C	G,D
    # data_type.append("fuse")
    # fuse_type.append("concat")
    # open_gan.append(True)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    # # M,L	MSA	G,D
    # data_type.append("fuse")
    # fuse_type.append("multi_modal_self_attn")
    # open_gan.append(True)
    # open_unmatch.append(False)
    # unmatch_k.append(16)
    
    
    # M,L	CA	G,C
    data_type.append("fuse")
    fuse_type.append("cross_attn")
    open_gan.append(False)
    open_unmatch.append(True)
    unmatch_k.append(16)
    # M,L	SA	G,C
    data_type.append("fuse")
    fuse_type.append("sep_attn")
    open_gan.append(False)
    open_unmatch.append(True)
    unmatch_k.append(16)
    # M,L	C	G,C
    data_type.append("fuse")
    fuse_type.append("concat")
    open_gan.append(False)
    open_unmatch.append(True)
    unmatch_k.append(16)
    # M,L	MSA	G,C
    data_type.append("fuse")
    fuse_type.append("multi_modal_self_attn")
    open_gan.append(False)
    open_unmatch.append(True)
    unmatch_k.append(16)
    
    
    # # M,L	CA	C,G,D
    # data_type.append("fuse")
    # fuse_type.append("cross_attn")
    # open_gan.append(True)
    # open_unmatch.append(True)
    # unmatch_k.append(16)
    # # M,L	SA	C,G,D
    # data_type.append("fuse")
    # fuse_type.append("sep_attn")
    # open_gan.append(True)
    # open_unmatch.append(True)
    # unmatch_k.append(16)
    # # M,L	C	C,G,D
    # data_type.append("fuse")
    # fuse_type.append("concat")
    # open_gan.append(True)
    # open_unmatch.append(True)
    # unmatch_k.append(16)
    # # M,L	MSA	C,G,D
    # data_type.append("fuse")
    # fuse_type.append("multi_modal_self_attn")
    # open_gan.append(True)
    # open_unmatch.append(True)
    # unmatch_k.append(16)
    
    temp2 = []
    params["window_size"] = 50
    params["hidden_size"] = 32
    for i in range(len(data_type)):
        params["data_type"] = data_type[i]
        params["fuse_type"] = fuse_type[i]
        params["open_gan"] = open_gan[i]
        params["open_unmatch_zoomout"] = open_unmatch[i]
        params["unmatch_k"] = unmatch_k[i]
        hash_id = f'{params["dataset"]}_{params["window_size"]}_{params["k"]}_{params["criterion"]}_{params["open_kpi_normalization"]}_{params["open_log_normalization"]}_{params["feature_type"]}_{params["open_min_max"]}_{params["open_kpi_select"]}_{params["kpi_ratio"]}_{params["kpi_with_high_std"]}_{params["attn_type"]}_{params["fuse_type"]}_{params["data_type"]}_{params["hidden_size"]}_{params["open_position_embedding"]}_{params["sigma_matrix"]}_{params["open_narrowing_modal_gap"]}_{params["open_feature2"]}_{params["open_expand_anomaly_gap"]}_{params["evaluation_sep"]}_{params["open_gan"]}_{params["open_unmatch_zoomout"]}_{params["open_gan_sep"]}_{params["unmatch_k"]}'
        res = caculate_mean_std_k("../result21/"+hash_id,0)
        temp1 = []
        temp1.append(res["f1"])
        temp1.append(res["rc"])
        temp1.append(res["pc"])
        temp2.append(temp1)
    res = np.array(temp2)
    res = np.around(res, 3)
    print(res)
    np.set_printoptions(suppress=True)
    # np.savetxt(f'../result21/{params["dataset"]}_ablation_res.csv', res, fmt="%.03f,%.03f,%.03f")
    np.savetxt(f'temp/{params["dataset"]}_ablation_res.csv', res, fmt="%.03f,%.03f,%.03f")
    # np.savetxt(f'../result21/{params["dataset"]}_ablation_res.csv', res, delimiter=",")

def visualization(data,target):
    # digits = load_digits()
    # data = digits.data[:,0].reshape(-1, 1)
    # target = digits.target
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(data)
    X_pca = PCA(n_components=2).fit_transform(data)

    ckpt_dir="temp"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target,label="t-SNE")
    plt.legend()
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target,label="PCA")
    plt.legend()
    plt.savefig(f'{ckpt_dir}/digits_tsne-pca.png', dpi=120)
    # plt.show()

def visualization_tsne(data,data2,data3,target,name):
    # digits = load_digits()
    # data = digits.data[:,0].reshape(-1, 1)
    # target = digits.target
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(data)
    X_tsne2 = TSNE(n_components=2,random_state=33).fit_transform(data2)
    X_tsne3 = TSNE(n_components=2,random_state=33).fit_transform(data3)
    
    # X_pca = PCA(n_components=2).fit_transform(data)
    # X_pca2 = PCA(n_components=2).fit_transform(data2)
    # X_pca3 = PCA(n_components=2).fit_transform(data3)

    ckpt_dir="temp"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.scatter(X_tsne[::10, 0], X_tsne[::10, 1], c=target,label="R")
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target,label="1")
    plt.legend()
    plt.subplot(132)
    plt.scatter(X_tsne2[::10, 0], X_tsne2[::10, 1], c=target,label="RA")
    # plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=target,label="2")
    plt.legend()
    plt.subplot(133)
    plt.scatter(X_tsne3[::10, 0], X_tsne3[::10, 1], c=target,label="RAC")
    # plt.scatter(X_pca3[:, 0], X_pca3[:, 1], c=target,label="3")
    plt.legend()
    plt.savefig(f'{ckpt_dir}/tsne_{name}.pdf')
    # plt.show()

def visualization_pca(data,data2,data3,target,name):
    # digits = load_digits()
    # data = digits.data[:,0].reshape(-1, 1)
    # target = digits.target
    # X_tsne = TSNE(n_components=2,random_state=33).fit_transform(data)
    # X_tsne2 = TSNE(n_components=2,random_state=33).fit_transform(data2)
    # X_tsne3 = TSNE(n_components=2,random_state=33).fit_transform(data3)
    
    X_pca = PCA(n_components=2).fit_transform(data)
    X_pca2 = PCA(n_components=2).fit_transform(data2)
    X_pca3 = PCA(n_components=2).fit_transform(data3)

    ckpt_dir="temp"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # k=29
    # k=9
    # k=5
    k = 9
    alpha_value = 0.5
    ymax = 1e-17
    ymin = -1e-17
    xmax = 0.4
    xmin = -0.05
    cc = ["m" if i==1 else "b" for i in target[::k]]
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target,label="1")
    plt.scatter(X_pca[::k, 0], X_pca[::k, 1], c=cc, alpha=alpha_value) # ,label="R"
    # plt.xticks([])
    # plt.yticks([])
    # plt.xlim(-0.1,0.4)
    plt.ylim([ymin,ymax])
    plt.xlim([xmin,xmax])
    plt.ylabel("Random Value")
    # plt.legend()
    # plt.title("Reconstruction learning")
    plt.title("Baseline")
    plt.subplot(132)
    # plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=target,label="2")
    plt.scatter(X_pca2[::k, 0], X_pca2[::k, 1], c=cc, alpha=alpha_value)
    # plt.xticks([])
    plt.yticks([])
    # plt.xlim(0,0.4)
    plt.ylim([ymin,ymax])
    plt.xlim([xmin,xmax])
    plt.xlabel("Reconstruction Error")
    # plt.legend()
    # plt.title("Adversarial reconstruction learning\n")
    plt.title("Adversarial Learning")
    plt.subplot(133)
    # plt.scatter(X_tsne3[:, 0], X_tsne3[:, 1], c=target,label="3")
    plt.scatter(X_pca3[::k, 0], X_pca3[::k, 1], c=cc, alpha=alpha_value)
    # plt.xticks([])
    plt.yticks([])
    plt.ylim([ymin,ymax])
    plt.xlim([xmin,xmax])
    # plt.legend()
    # plt.title("Adversarial contrastive \nreconstruction learning")
    plt.title("Adversarial Contrastive Learning")
    # fig,axes = plt.subplots(3,2,sharey=True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0)
    plt.savefig(f'{ckpt_dir}/pca_{name}.pdf')
    # plt.show()

def visualization_distance(data,data2,data3,target,name,type_name):    
    X_pca = data
    X_pca2 = data2
    X_pca3 = data3
    cs = ["b","m"]
    alpha_value = 0.5

    k= 3
    # cc = [ cs[i] for i in target[::k]]
    cc = ["m" if i==1 else "b" for i in target[::k]]
    # for i in :
    #     cc.append(cs[i])
    # xmax = 10e-7
    # xmin = -10e-7
    xmax = 0.4
    xmin = -0.01
    plt.figure(figsize=(10, 5))
    plt.subplot(131)
    plt.scatter(X_pca[::k, 0], X_pca[::k, 1], c=cc, alpha=alpha_value) # ,label="R"
    plt.ylabel("Random Value")
    plt.title("Baseline")
    plt.xlim([xmin,xmax])
    plt.xlabel(type_name)
    plt.ticklabel_format(style='sci',scilimits=(-1,2),  axis='x')
    
    
    plt.subplot(132)
    plt.scatter(X_pca2[::k, 0], X_pca2[::k, 1], c=cc, alpha=alpha_value)
    plt.xlim([xmin,xmax])
    plt.ticklabel_format(style='sci',scilimits=(-1,2),  axis='x')
    plt.yticks([])
    plt.xlabel(type_name)
    plt.title("Adversarial Learning")
    
    
    plt.subplot(133)
    plt.scatter(X_pca3[::k, 0], X_pca3[::k, 1], c=cc, alpha=alpha_value)
    plt.xlim([xmin,xmax])
    plt.ticklabel_format(style='sci',scilimits=(-1,2), axis='x')
    plt.yticks([])
    plt.title("Adversarial Collaborative Learning")
    
    
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.1, hspace=0)
    plt.savefig(f'temp/distance_{name}.pdf')


def visualization_(): 
    # dataset = "zte"
    # dataset = "yzh"
    dataset = "yzh"
    # embeds = np.load(f"temp/embeds_{dataset}_False_False_16.npy")
    # embeds2 = np.load(f"temp/embeds_{dataset}_True_False_16.npy")
    # embeds3 = np.load(f"temp/embeds_{dataset}_True_True_16.npy")
    distances = np.load(f"temp/distance_{dataset}_False_False_16.npy").reshape(-1, 1).repeat(2,axis=-1)
    distances2 = np.load(f"temp/distance_{dataset}_True_False_16.npy").reshape(-1, 1).repeat(2,axis=-1)
    distances3 = np.load(f"temp/distance_{dataset}_True_True_16.npy").reshape(-1, 1).repeat(2,axis=-1) 
    labels = np.load(f"temp/labels_{dataset}.npy")
    visualization_pca(distances,distances2,distances3,labels,f"{dataset}_distances")
    
    
    # basedistances = np.load(f"temp/distance_{dataset}_True_True_16.npy").reshape(-1)
    # index_ = np.argsort(basedistances)
    # length = int(0.0*len(basedistances)) 
    # distances = np.load(f"temp/distance_{dataset}_False_False_16.npy").reshape(-1, 1)[index_[length:]].repeat(2,axis=-1)
    # distances2 = np.load(f"temp/distance_{dataset}_True_False_16.npy").reshape(-1, 1)[index_[length:]].repeat(2,axis=-1)
    # distances3 = np.load(f"temp/distance_{dataset}_True_True_16.npy").reshape(-1, 1)[index_[length:]].repeat(2,axis=-1) 
    # labels = np.load(f"temp/labels_{dataset}.npy")[index_[length:]]
    # visualization_pca(distances,distances2,distances3,labels,f"{dataset}_distances")
    
    # distances = np.load(f"temp/distance_{dataset}_False_False_16.npy").reshape(-1, 1)
    # distances2 = np.load(f"temp/distance_{dataset}_True_False_16.npy").reshape(-1, 1)
    # distances3 = np.load(f"temp/distance_{dataset}_True_True_16.npy").reshape(-1, 1)
    # x = np.random.normal(loc=0.0, scale=1e-5, size=distances.shape)
    # distances  = np.concatenate([distances,x],axis=-1)
    # distances2  = np.concatenate([distances2,x],axis=-1)
    # distances3  = np.concatenate([distances3,x],axis=-1)
    # typename = "Reconstruction Error"
    # visualization_distance(distances,distances2,distances3,labels,f"{dataset}_{typename}",typename)
    
        
    # visualization_tsne(embeds,embeds2,embeds3,labels,f"{dataset}_embeds")
    # visualization_pca(embeds,embeds2,embeds3,labels,f"{dataset}_embeds")

    # visualization_tsne(distances,distances2,distances3,labels,f"{dataset}_distances")

if __name__ == "__main__":
    # visualization_()
    # plot_unmatched_k()
    plot_unmatched_k_v2()
    # plot_hidden_size()
    # plot_window_size()
    # plot_kpi_ratio_and_kpi_with_high_std()
    # caculate_mean_std_k("../result2/original_1_l1_True_False_template_appear_False_False_40_False_add_cross_attn_fuse_64_False_False_False_False_False_False",20)
    # caculate_mean_std_k("../result2/original_l1_True_False_log_64_False",20)
    # caculate_mean_std_k("../result21/original_30_1_l1_True_False_template_appear_False_False_40_False_add_concat_fuse_64_False_False_True_False_True_False_True_True_False_4",0)
    # caculate_mean_std_k("../result21/zte_30_1_l1_True_False_template_appear_False_False_40_False_add_multi_modal_self_attn_kpi_64_False_False_True_False_True_False_False_False_False_1",0)
    # caculate_mean_std_k("../result21/zte_30_1_l1_True_False_template_appear_False_False_40_False_add_multi_modal_self_attn_fuse_64_False_False_True_False_True_False_False_False_False_1",0)
    # caculate_mean_std_k("../result21/zte_30_1_l1_True_False_template_appear_False_False_40_False_add_multi_modal_self_attn_log_64_False_False_True_False_True_False_True_False_False_1",0)
    # display_ablation()
    # params = ["dataset","window_size","k","criterion",
    #                     "open_kpi_normalization","open_log_normalization",
    #         "feature_type",
    #                     "open_min_max","open_kpi_select",
    #         "kpi_ratio",
    #                     "kpi_with_high_std",
    #         "attn_type","fuse_type","data_type","hidden_size",
    #                     "open_position_embedding","sigma_matrix","open_narrowing_modal_gap","open_feature2","open_expand_anomaly_gap","evaluation_sep","open_gan","open_unmatch_zoomout","open_gan_sep",
    #         "unmatched_k","run_times"]
    # params = ["dataset", "criterion", 
    #                     "open_kpi_normalization", "open_log_normalization", 
    #         "data_type", "hidden_size", 
    #                     "open_position_embedding", "open_gan", 
    #         "run_times"]

        