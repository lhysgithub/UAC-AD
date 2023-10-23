import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import json
import pandas as pd

def find_positive_segment(data):
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

def plot_curve(data,name):
    l , k = data.shape
    for i in range(k):
        plt.cla()
        x = range(len(data))
        plt.plot(x, data[:,i], "r", label=f"{i}_{name}",alpha=0.5)
        plt.legend()
        plt.savefig(f"temp/{i}_{name}.pdf")

def plot_data(data,name):
    kpis = []
    logs = []
    labels = []
    log_raws = []
    cnt = 0
    names = []
    for k,v in data.items():
        if cnt ==0:
            names = v["metric_name"]
        cnt+=1
        kpis.append(v["kpis"])
        labels.append(v["label"])
        logs.append(v["logs"])
        log_raws.append(v["log"])
    kpis = np.array(kpis)
    kpis = np.asarray(kpis, dtype=np.float32)
    # kpis, scaler = normalize_data(kpis, scaler=None)
    # plot_curve(kpis,name)
    # index_list, lens_list = find_positive_segment(labels)
    # print(index_list)
    # print(lens_list)
    plot_labeled_curve_with_names(kpis,labels,name,names)

def plot_labeled_curve(data,labels,name):
    l , k = data.shape
    index_list, lens_list = find_positive_segment(labels)
    for i in range(k):
        # if i !=106:
        #     continue
        plt.cla()
        x = range(len(data))
        plt.plot(x, data[:,i], "b", label=f"{i}_{name}",alpha=0.5)
        for j in range(len(index_list)):
            start = index_list[j]
            end = index_list[j] + lens_list[j]
            y_min = data[:,i].min()
            y_max = data[:,i].max()
            plt.fill_between([start, end], y_min, y_max, facecolor='pink', alpha=0.8)
        plt.legend()
        plt.savefig(f"temp2/{i}_{name}.pdf")

def plot_labeled_curve_with_names(data,labels,name,names):
    l , k = data.shape
    index_list, lens_list = find_positive_segment(labels)
    for i in range(k):
        # if i !=106:
        #     continue
        plt.cla()
        x = range(len(data))
        plt.plot(x, data[:,i], "b", label=f"{names[i]}",alpha=0.5)
        for j in range(len(index_list)):
            start = index_list[j]
            end = index_list[j] + lens_list[j]
            y_min = data[:,i].min()
            y_max = data[:,i].max()
            plt.fill_between([start, end], y_min, y_max, facecolor='pink', alpha=0.8)
        plt.legend()
        plt.savefig(f"temp2/{i}_{name}.pdf")

def add_anomalies(train_data,target_rate):
    temp_data = []
    labels = []
    for idx,dic in enumerate(train_data.items()) :
        temp_data.append(dic)
        labels.append(dic[1]["label"])
    index_list, lens_list = find_positive_segment(labels)
    anomaly_rate = sum(lens_list) / len(labels)
    print(anomaly_rate)
    # target_rate= sum(lens_list) +  wait_to_add_anomaly/ len(labels) + wait_to_add_anomaly
    # target_rate*(len(labels) + wait_to_add_anomaly) = sum(lens_list) +  wait_to_add_anomaly
    # target_rate*len(labels) -  sum(lens_list) = wait_to_add_anomaly *(1-target_rate)
    wait_to_add_anomaly = int((target_rate*len(labels) -  sum(lens_list)) / (1-target_rate))
    added_anomaly = 0
    while added_anomaly <wait_to_add_anomaly:
        new_id = np.random.randint(len(index_list)+1)-1
        base = index_list[new_id]
        anomalies = lens_list[new_id]
        for i in range(anomalies):
            train_data[f"repeat_anomaly_{added_anomaly}_{temp_data[base][0]}"] = temp_data[base][1]
            added_anomaly += 1

    temp_data = []
    labels = []
    for idx,dic in enumerate(train_data.items()) :
        temp_data.append(dic)
        labels.append(dic[1]["label"])
    index_list, lens_list = find_positive_segment(labels)
    anomaly_rate = sum(lens_list) / len(labels)
    print(anomaly_rate)
    return train_data

def dump_logs_template_seq(data,name):
    log_templates = []
    labels = []
    logs_ = []
    for k,v in data.items():
        labels.append(v["label"])
        logs_.append(v["logs"])
        for log in v["logs"]:
            if log not in log_templates:
                log_templates.append(log)
    logs = []
    normal_logs = []
    anomaly_logs = []
    anomaly = False
    index_list, lens_list = find_positive_segment(labels)
    for i in range(len(logs_)):
        for j in range(len(index_list)):
            index = index_list[j]
            lens = lens_list[j]
            if i >= index and i< index + lens:
                anomaly = True
                break
        
        session_logs = []
        for log in logs_[i]:
            log_template_idx = log_templates.index(log)
            session_logs.append(str(log_template_idx))

        if anomaly:
            anomaly_logs.append(" ".join(session_logs))
        else:
            normal_logs.append(" ".join(session_logs))
        logs.append(" ".join(session_logs))
        anomaly = False
                
    
    # for k,v in data.items():
    #     session_logs = []
    #     for log in v["logs"]:
    #         log_template_idx = log_templates.index(log)
    #         session_logs.append(str(log_template_idx))
    #     logs.append(" ".join(session_logs))
    normal_logs = "\n".join(normal_logs)
    anomaly_logs = "\n".join(anomaly_logs)
    logs = "\n".join(logs)
    with open(f"data_for_deeplog/{name}.txt","w") as f:
        f.write(logs)
    with open(f"data_for_deeplog/{name}_normal.txt","w") as f:
        f.write(normal_logs)
    with open(f"data_for_deeplog/{name}_anormaly.txt","w") as f:
        f.write(anomaly_logs)
    # logs = np.array(logs)
    # np.savetxt(name,logs,delimiter=" ")         

def dump_kpis(data,name):
    kpis = []
    labels = []
    dataset_name = name.split("_")[0]
    for k,v in data.items():
        if dataset_name == "hades":
            kpis.append(v["kpis"].mean(-2))
        else:
            kpis.append(v["kpis"])
        labels.append(v["label"])
    kpis = np.array(kpis)
    labels = np.array(labels)
    res_pd = pd.DataFrame(kpis)
    res_pd.index.name = "timestamp"
    res_pd.to_csv(f"data_for_scwarn/{name}.csv")
    labels_pd = pd.DataFrame(labels)
    labels_pd.to_csv(f"data_for_scwarn/label_{name}.csv")
    # np.savetxt(name,kpis,delimiter=",") 

def dump_kpis_for_TSAD(data,name):
    kpis = []
    labels = []
    dataset_name = name.split("_")[0]
    for k,v in data.items():
        if dataset_name == "hades":
            kpis.append(v["kpis"].mean(-2))
        else:
            kpis.append(v["kpis"])
        labels.append(v["label"])
    kpis = np.array(kpis)
    labels = np.array(labels)
    np.save(f"data_for_tsad/{name}.npy",kpis) 
    np.save(f"data_for_tsad/label_{name}.npy",labels) 
    
    # res_pd = pd.DataFrame(kpis)
    # res_pd.index.name = "timestamp"
    # res_pd.to_csv(f"{name}.csv")
    # labels_pd = pd.DataFrame(labels)
    # labels_pd.to_csv(f"label_{name}.csv")
    # np.savetxt(name,kpis,delimiter=",") 

def get_dataset_distribution(data):
    temp_data = []
    labels = []
    logs = []
    kpis = []
    log_raw = []
    for idx,dic in enumerate(data.items()) :
        temp_data.append(dic)
        labels.append(dic[1]["label"])
        logs.append(dic[1]["logs"])
        kpis.append(dic[1]["kpis"])
        log_raw.append(dic[1]["log"])
    log_length = 0
    for log in logs:
        log_length += len(log)
    kpi_length = len(kpis)
    index_list, lens_list = find_positive_segment(labels)
    anomaly_rate = sum(lens_list) / len(labels)
    # print(anomaly_rate)
    return {"anomaly_rate":anomaly_rate,"log_length":log_length,"kpi_length":kpi_length,"anomaly_timestamps":index_list,"anomaly_duration":lens_list}

def isInAnomalyAround(cnt,index_list3,around):
    for i in index_list3:
        if cnt< i + around and cnt >= i-around:
            return True
    return False

def isInAnomalyAroundForSupervised(cnt,index_list3,around):
    around = 10
    for i in index_list3:
        if cnt< i + around and cnt >= i-around:
            return True
    return False

def isInAnomalyAround2(cnt,index_list2,around,k):
    for i in index_list2:
        # if i == index_list2[-1]:
        #     if  cnt< i + around*k and cnt >= i:
        #         return True
        # el
        if cnt< i + around*k and cnt >= i-around*k:
            return True
    return False
    
def patching_kpi():
    data_dir = "../../data/zte/"
    # save_dir = "../../data/zte2/"
    
    # q: node 指标是不是覆盖了？
    # q: 需要加上disk partition的指标
    # 需要打标
    # 需要平衡分布
    # 
    
    with open(data_dir+"metric_data.json") as f:
        metrics = json.load(f)
    
    stamp_length = 0
    complete_m = ""
    for k,v in metrics.items():
        if stamp_length < len(v):
            stamp_length = len(v)
            complete_m = k
    
    uncomplete_ms = []
    complete_ms = []
    for k,v in metrics.items():
        if len(v) < stamp_length:
            uncomplete_ms.append(k)
        else:
            complete_ms.append(k)
            
    stamps = [] 
    for k,v in metrics[complete_m].items():
        stamps.append(k)      
    
    for k in stamps:
        for name in uncomplete_ms:
            if k not in metrics[name].keys():
                metrics[name][k] = "0"
        
    with open(data_dir+"metric_data_.json","w") as f:
        json.dump(metrics,f)
# if __name__ == "__main__":
    
    