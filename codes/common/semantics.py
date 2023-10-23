'''
Be aware that a log sequence inside a chunk has been padded.
'''
import math
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
def tokenize(log):
    word_lst_tmp = re.findall(r"[a-zA-Z]+", log)
    word_lst = []
    for word in word_lst_tmp:
        res = list(filter(None, re.split("([A-Z][a-z][^A-Z]*)", word)))
        if len(res) == 0: word_lst.append(word.lower())
        else: word_lst.extend([w.lower() for w in res])
    return word_lst

from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
import os
class Vocab:
    def __init__(self, **kwargs):
        self.embedding_dim = kwargs["word_embedding_dim"]
        self.save_dir = kwargs["word2vec_save_dir"]
        self.model_type = kwargs["word2vec_model_type"]
        self.epochs = kwargs["word2vec_epoch"]
        self.word_window = kwargs["word_window"]
        self.log_lenth = 0
        self.save_path = os.path.join(self.save_dir, self.model_type+"-"+str(self.embedding_dim)+".model")

    def get_word2vec(self, logs):
        if os.path.exists(self.save_path): #load
            if self.model_type == "naive" or self.model_type == "skip-gram": 
                model = Word2Vec.load(self.save_path)
            elif self.model_type == "fasttext": 
                model = FastText.load(self.save_path)
        else:
            ####### Load log corpus #######
            sentences = [["padding"]]
            for log in logs:
                word_lst = tokenize(log)
                if len(set(word_lst)) == 1 and word_lst[0] == "padding": continue
                self.log_lenth = max(self.log_lenth, len(word_lst))
                sentences.append(word_lst)
            
            ####### Build model #######
            if self.model_type == "naive": 
                model = Word2Vec(window=self.word_window, min_count=1, vector_size=self.embedding_dim)
            elif self.model_type == "skip-gram": 
                model = Word2Vec(sg=1, window=self.word_window, min_count=1, vector_size=self.embedding_dim)
            elif self.model_type == "fasttext": 
                model = FastText(window=self.word_window, min_count=1, vector_size=self.embedding_dim)
            model.build_vocab(sentences)
            
            ####### Train and Save#######
            model.train(sentences, total_examples=len(sentences), epochs=self.epochs)
            os.makedirs(self.save_dir, exist_ok=True)
            model.save(self.save_path)

        self.word2vec = model
        self.wv = model.wv; del model.wv

from sklearn.base import BaseEstimator   
import numpy as np
import logging    
import itertools

class FeatureExtractor(BaseEstimator):
    def __init__(self, **kwargs):
        self.feature_type = kwargs["feature_type"]
        self.data_type = kwargs["data_type"]
        # self.log_window_size = kwargs["log_window_size"]
        self.model_type = kwargs["word2vec_model_type"]
        self.embedding_dim = kwargs["word_embedding_dim"]
        self.vocab = Vocab(**kwargs)
        self.meta_data = {"num_labels":2, "max_log_lenth":1}
        self.oov = set()
    

    def __log2vec(self, log):
        if log=="padding":
            return np.array(np.zeros(32)).astype("float32")
        word_lst = tokenize(log)
        feature = []
        for word in word_lst:
            if word in self.known_words:
                feature.append(self.word_vectors[word]*self.idf[word]/len(word_lst))
            else:
                self.oov.add(word)
                if self.model_type == "naive" or self.model_type == "skip-gram": 
                    feature.append(np.random.rand(self.word_vectors["padding"].shape)-0.5) #[-0.5, 0.5]
                else: 
                    feature.append(self.word_vectors[word]*self.idf[word]*1/len(word_lst))
        return np.array(feature).sum(axis=0).astype("float32") #[embedding_dim]
        
    def __seqs2feat(self, seqs):
        #seqs in a chunk, with the number of chunk_length
        if self.feature_type == "word2vec":
            # return np.array([[self.__log2vec(log) for log in seq] for seq in seqs]).astype("float32")
            return np.array([self.__log2vec(log) for log in seqs[:60]]).astype("float32")

        if self.feature_type == "sequential":
            return np.array([[self.log2id_train.get(log, 1)  for log in seq] for seq in seqs]).astype("float32")
        if self.feature_type=="template_appear":
            # template_count=[0]*(1+self.cluster_y_pred.max())
            template_count=[0]*(len(self.log2id_train))
            # template_count=[0]*(len(self.log2id_train)-1)
            # padding_index=self.log2id_train["padding"]
            for log in seqs:

                # if log== 'padding':
                #     continue
                try:
                    idx=self.log2id_train[log]-1
                    # idx=self.log2id_train[log]
                    template_count[idx] = 1
                except Exception as e:
                    continue
                # template_count[self.cluster_y_pred[idx]]+=1
            return np.array(template_count)

            # if np.array(template_count).sum()>0:
            #     template_count=np.array(template_count)/np.array(template_count).sum()
            # return np.around(np.array(template_count), 2)
        if self.feature_type=="template_count":
            # template_count=[0]*(1+self.cluster_y_pred.max())
            template_count=[0]*(len(self.log2id_train))
            # padding_index=self.log2id_train["padding"]
            for log in seqs:

                # if log== 'padding':
                #     continue
                try:
                    idx=self.log2id_train[log]-1
                    template_count[idx] += 1
                except Exception as e:
                    break
                # template_count[self.cluster_y_pred[idx]]+=1
            return np.array(template_count)

    def fit(self, chunks):
        total_logs = list(itertools.chain(*[v["logs"] for _, v in chunks.items()]))
        self.ulog_train = set(total_logs)
        if "padding" in self.ulog_train:
            self.ulog_train.remove("padding") # for data3
        # self.id2log_train = {0: "oovlog"}
        self.id2log_train={}
        self.id2log_train.update({idx: log for idx, log in enumerate(self.ulog_train, 1)})
        self.log2id_train = {v: k for k, v in self.id2log_train.items()}

        if self.feature_type == "word2vec"or "template_count":
            self.vocab.get_word2vec(total_logs)
            self.word_vectors = self.vocab.wv
            self.known_words = self.vocab.wv.key_to_index #known word list
            self.meta_data["vocab_size"] = len(self.known_words)
            self.meta_data["max_log_lenth"] = self.vocab.log_lenth if self.vocab.log_lenth>0 else 50
            logging.info("{} tempaltes are found.".format(len(self.log2id_train) - 1))
            template = list(self.log2id_train.keys())

            tem_list=[tokenize(t) for t in template]
            self.idf={}
            for tem in tem_list:
                for word in tem:
                    if word not in self.idf.keys():
                        self.idf[word]=1
                    else:
                        self.idf[word]+=1
            for k,v in self.idf.items():
                self.idf[k]=math.log(len(tem_list)/(v+1))

            # template_feature = np.array([self.__log2vec(t) for t in template])
            # kmean = KMeans(n_clusters=10)
            # self.cluster_y_pred = kmean.fit_predict(template_feature)
        elif self.feature_type == "sequential" :
            self.meta_data["vocab_size"] = len(self.log2id_train)
        
        else: raise ValueError("Unrecognized feature type {}".format(self.feature_type))


    def transform(self, chunks,datatype="train"):
        logging.info("Transforming {} data.".format(datatype))
        
        if not ("train" in datatype): # handle new logs
            total_logs = list(itertools.chain(*[v["logs"] for _, v in chunks.items()]))
            ulog_new = set(total_logs) - self.ulog_train
            logging.info(f"{len(ulog_new)} new templates show.")
            # for u in ulog_new: print(u)
        
        for id, item in tqdm(chunks.items()):
            chunks[id]["log_features"] = self.__seqs2feat(item["logs"])
            
        if len(self.oov) > 0: 
            logging.info("{} OOV words: {}".format(len(self.oov), ",".join(list(self.oov))))
        
        return chunks
                
    
        

