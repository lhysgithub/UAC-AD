import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import json
from data_processing_utils import *

def build_multi_modal_data_zte():
    data_dir = "../../data/zte/"
    save_dir = "../../data/zte2/"
    
    # with open(data_dir+"metric_data.json") as f:
    #     metrics = json.load(f)

    # all_data = {}
    train_data = {}
    test_data = {}
    cnt = 0
    target_rate = 0.3
    # index_list = [252, 281, 290, 312, 342, 360, 371, 378, 414, 654, 696, 706, 713, 2052, 4038, 4064, 4084, 4711, 4806, 7084]
    # index_list2 = [4041, 4711, 7084]
    # index_list3 = [252, 281, 290, 312, 342, 371, 378, 414]
    # around = 120
    # k = 2.2
    # split_ = 500
    # start_ = 4000
    # end_ = 5000

    for filename in os.listdir(data_dir):
        kpis = 0
        if filename.endswith(".pkl"):
            with open(os.path.join(data_dir, filename), "rb") as fr:
                temp_data = pickle.load(fr)
                original_data = temp_data
                res = get_dataset_distribution(original_data)
                anomaly_timestamps = res["anomaly_timestamps"]
                # around = 300
                around = 100
                print("zte "+'\n'.join(["{}:{}".format(k, v) for k,v in res.items()])+'\n\n')
                plot_data(original_data,f"zte_original_all_data")
                
                
                length = int(len(temp_data)*1)
                cnt = 0
                for key_, value_ in temp_data.items():
                    kpis = len(value_["kpis"])
                    new_value = value_
                    # labels.append(new_value["label"])
                    if isInAnomalyAround(cnt,anomaly_timestamps,around):
                    # if isInAnomalyAroundForSupervised(cnt,anomaly_timestamps,around):
                        test_data[f"{key_}_{filename}"] = new_value
                    else:
                        train_data[f"{key_}_{filename}"] = new_value
                        
                    # elif isInAnomalyAround2(cnt):
                    #     train_data[f"{key_}_{filename}"] = new_value
                    # if cnt<split_:
                    #     test_data[f"{key_}_{filename}"] = new_value
                    # elif start_<cnt < end_:
                    #     train_data[f"{key_}_{filename}"] = new_value
                    # if cnt < int(0.5*length):
                    #     train_data[f"{key_}_{filename}"] = new_value
                    # elif int(0.5*length) < cnt < length:
                    #     test_data[f"{key_}_{filename}"] = new_value
                    cnt += 1
        print(f"{kpis}_{filename}")
        
    all_data = {}
    all_data.update(train_data)
    all_data.update(test_data)
    res = get_dataset_distribution(all_data)
    print("zte "+'\n'.join(["{}:{}".format(k, v) for k,v in res.items()])+'\n\n')
    print(f"train set: {len(train_data)} test set: {len(test_data)}")

    # # DeepLog
    # dump_logs_template_seq(train_data,"zte_log_train")
    # dump_logs_template_seq(test_data,"zte_log_test")
    # dump_logs_template_seq(all_data,"zte_log")
    
    # # SCWarn
    # dump_kpis(train_data,"zte_kpi_train")
    # dump_kpis(test_data,"zte_kpi_test")
    
    # omni and mtad-gat
    dump_kpis_for_TSAD(train_data,"zte_kpi_train")
    dump_kpis_for_TSAD(test_data,"zte_kpi_test")

    # # hades and logRobust
    # train_data = add_anomalies(train_data,target_rate)
    # save_dir = "data_for_hades"
    # all_data = {}
    # all_data.update(train_data)
    # all_data.update(test_data)
    # res = get_dataset_distribution(all_data)
    # print("zte "+'\n'.join(["{}:{}".format(k, v) for k,v in res.items()])+'\n\n')
    
    # others our
    # plot_data(all_data,f"zte_all_data")
    
    with open(os.path.join(save_dir, "train.pkl"), "wb") as fw:
        pickle.dump(train_data,fw)
    with open(os.path.join(save_dir, "unlabel.pkl"), "wb") as fw:
        pickle.dump(train_data,fw)
    with open(os.path.join(save_dir, "test.pkl"), "wb") as fw:
        pickle.dump(test_data,fw)

def build_multi_modal_data_yzh():
    data_dir = "../../data/data2/"
    save_dir = "../../data/data3/"
    train_data = {}
    test_data = {}
    cnt = 0
    target_rate = 0.3
    for filename in os.listdir(data_dir):
        kpis = 0
        if filename.endswith(".pkl"):
            with open(os.path.join(data_dir, filename), "rb") as fr:
                temp_data = pickle.load(fr)
                length = len(temp_data)
                cnt = 0
                for key_, value_ in temp_data.items():
                    kpis = len(value_["kpis"])
                    if kpis != 159:
                        break
                    new_value = value_
                    if cnt < int(0.7*length):
                        train_data[f"{key_}_{filename}"] = new_value
                    else:
                        test_data[f"{key_}_{filename}"] = new_value
                    cnt += 1
        print(f"{kpis}_{filename}")
    
    all_data = {}
    all_data.update(train_data)
    all_data.update(test_data)
    res = get_dataset_distribution(all_data)
    print("yzh "+'\t'.join(["{}:{:.4f}".format(k, v) for k,v in res.items()])+'\n\n')

    # DeepLog
    # dump_logs_template_seq(train_data,"temp/yzh_log_train")
    # dump_logs_template_seq(test_data,"temp/yzh_log_test")
    # dump_logs_template_seq(all_data,"temp/yzh_log")
    
    # SCWarn
    # dump_kpis(train_data,"yzh_kpi_train")
    # dump_kpis(test_data,"yzh_kpi_test")
    
    # omni and mtad-gat
    dump_kpis_for_TSAD(train_data,"yzh_kpi_train")
    dump_kpis_for_TSAD(test_data,"yzh_kpi_test")

    # hades and logRobust
    train_data = add_anomalies(train_data,target_rate)

    # others our

    print(len(train_data))
    print(len(test_data))
    
    with open(os.path.join(save_dir, "train.pkl"), "wb") as fw:
        pickle.dump(train_data,fw)
    with open(os.path.join(save_dir, "unlabel.pkl"), "wb") as fw:
        pickle.dump(train_data,fw)
    with open(os.path.join(save_dir, "test.pkl"), "wb") as fw:
        pickle.dump(test_data,fw)
        
def build_multi_modal_data_hades():
    data_dir = "../../data/chunk_10/"
    all_data = {}
    with open(os.path.join(data_dir, "train.pkl"), "rb") as fr:
        train_data = pickle.load(fr)
    with open(os.path.join(data_dir, "unlabel.pkl"), "rb") as fr:
        unlabel_data = pickle.load(fr)
    with open(os.path.join(data_dir, "test.pkl"), "rb") as fr:
        test_data = pickle.load(fr)
    all_data.update(train_data)
    all_data.update(unlabel_data)
    all_data.update(test_data)
    res = get_dataset_distribution(all_data)
    print("hades "+'\t'.join(["{}:{:.4f}".format(k, v) for k,v in res.items()])+'\n\n')
    
    # DeepLog
    # dump_logs_template_seq(train_data,"temp/hades_log_train")
    # dump_logs_template_seq(test_data,"temp/hades_log_test")
    # dump_logs_template_seq(all_data,"temp/hades_log")
    
    # SCWarn
    # dump_kpis(train_data,"hades_kpi_train")
    # dump_kpis(test_data,"hades_kpi_test")
    
    # omni and mtad-gat
    dump_kpis_for_TSAD(train_data,"hades_kpi_train")
    dump_kpis_for_TSAD(test_data,"hades_kpi_test")
    
    
if __name__ == "__main__":
    build_multi_modal_data_zte()
    # build_multi_modal_data_yzh()
    # build_multi_modal_data_hades()
    # patching_kpi()
    