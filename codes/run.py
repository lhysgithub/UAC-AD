from torch.utils.data import DataLoader
from common.data_loads import load_sessions, Process
from common.utils import *
import torch
import logging
from tqdm import tqdm
from models.basev3 import BaseModel
from common.data_processing_utils import *
# from models.basev4 import BaseModel
# from models.base import *

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
parser.add_argument("--patience", default=5, type=int) # 10 for zte
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
# parser.add_argument("--open_narrowing_modal_gap", default=False, type=str2bool) 
parser.add_argument("--open_narrowing_modal_gap", default=True, type=str2bool) # True for hades
parser.add_argument("--open_feature2", default=False, type=str2bool)
# parser.add_argument("--open_expand_anomaly_gap", default=False, type=str2bool) 
parser.add_argument("--open_expand_anomaly_gap", default=True, type=str2bool) # True for hades
parser.add_argument("--evaluation_sep", default=False, type=str2bool)
parser.add_argument("--open_gan_sep", default=False, type=str2bool)
parser.add_argument("--open_gan", default=True, type=str2bool)
parser.add_argument("--open_unmatch_zoomout", default=True, type=str2bool)
# parser.add_argument("--unmatch_k", default=16, type=int)
parser.add_argument("--unmatch_k", default=16, type=int)
parser.add_argument("--run_times", default=0, type=int)
parser.add_argument("--theta", default=0.15, type=float) # 0.3 0.15
parser.add_argument("--anomaly_rate", default=20, type=int) # 20 10 10
parser.add_argument("--criterion", default="l1", type=str, choices=["l1", "mse"])


##### Munual params
parser.add_argument("--window_size", default=50, type=int)
parser.add_argument("--hidden_size", default=32, type=int, help="Dim of the commnon feature space") # 可调
# Fuse params
parser.add_argument("--data_type", default="kpi", choices=["fuse", "log", "kpi"])
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
# params["hash_id"] = dump_params(params)
seed_everything(params["random_seed"])
os.environ ["CUDA_VISIBLE_DEVICES"] = params["gpu_device"]

if params["gpu"] and torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Using GPU...")
else:
    device = torch.device("cpu")
    logging.info("Using CPU...")

def main(var_nums):
    logging.info("^^^^^^^^^^ Current Model:"+params["main_model"]+", "+str(params["hash_id"])+" ^^^^^^^^^^")

    ###### Load data  ######
    train_chunks, unlabel_chunks, test_chunks = load_sessions(data_dir=params["data"],**params)
        
    unsupervised_chunks={}
    for key, value in train_chunks.items():
        if value["label"]==0:
            unsupervised_chunks[key]=value
    for key, value in unlabel_chunks.items():
        if value["label"]==0:
            unsupervised_chunks[key]=value
            
    # all_chunks = {}
    # all_chunks.update(train_chunks)
    # all_chunks.update(test_chunks)
    # res = get_dataset_distribution(all_chunks)
    # print("zte "+'\n'.join(["{}:{}".format(k, v) for k,v in res.items()])+'\n\n')
    

    if params["supervised"]:
        train_chunks.update(unlabel_chunks)
    processed = Process(var_nums, train_chunks,unlabel_chunks,unsupervised_chunks, test_chunks, **params)

    for key, value in processed.test_chunks.items():
        params["kpi_c"] = len(value["kpis"])
        params["log_c"] = len(value["log_features"])
        break

    print(params)
    bz = params["batch_size"]
    # train_loader = DataLoader(processed.dataset["train"], batch_size=bz, shuffle=True, pin_memory=True)
    unlabel_loader = DataLoader(processed.dataset["unlabel"],batch_size=bz, shuffle=True, pin_memory=True)
    test_loader = DataLoader(processed.dataset["test"], batch_size=bz, shuffle=False, pin_memory=True)

    ##### Build/Train model #####
    if params['data_type'] != 'kpi':
        vocab_size = processed.ext.meta_data["vocab_size"]
        logging.info("Known word number: {}".format(vocab_size))
    else: vocab_size=300
    model = BaseModel(device=device, var_nums=var_nums, vocab_size=vocab_size, **params)

    if params["pre_model"] is None: #train
        if params["supervised"]:
            # scores = model.supervised_fit(train_loader, test_loader)
            pass
        else:
            scores = model.unsupervised_fit(unlabel_loader, test_loader)
    else:
        model.load_model(params["pre_model"])
        scores = model.evaluate(test_loader)
    
    ##### Record results #####
    dump_scores(params["result_dir"], params["hash_id"], scores, model.train_time)
    logging.info("Current hash id {}".format(params["hash_id"]))

for run_times in range(0,5):
    params["run_times"] = run_times
    if params["dataset"] == 'yzh':
        params["open_kpi_select"] = False
    else:
        params["open_kpi_select"] = False
    params["hash_id"] = dump_params(params)
    main([4,3,2,2])
