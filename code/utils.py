#coding:utf8
# This file defines some helper functions.

import torch
import pickle
import torch.nn as nn
from sklearn import preprocessing
from torch.nn import Parameter,Module
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import pprint,copy,os,random,math,sys,pickle,time
import numpy as np
import networkx as nx
torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
# from whim_common.utils.progress import get_progress_bar
# use_cuda = False
from multiprocessing import Process,Pool

verb_net3_mapping_with_args='../data/encoding_with_args.csv'


def trans_to_cuda(variable):
    if use_cuda:
        return variable.cuda()    
    else:
        return variable

def id_to_vec(emb_file):
    dic={}
    for s in open(emb_file):
        s=s.strip().split()
        if len(s)==2:
            continue
        dic[s[0]]=np.array(s[1:],dtype=np.float32)
    dic['0']=np.zeros(len(dic['0']),dtype=np.float32)
    return dic

def word_to_id(voc_file):
    dic={}
    for s in open(voc_file):
        s=s.strip().split()
        dic[s[1]]=s[0]
    return dic

def get_word_vec(id_vec):
    word_vec=[]
    for i in range(len(id_vec)):
        word_vec.append(id_vec[str(i)])
    return np.array(word_vec,dtype=np.float32)

def get_hash_for_word(emb_file,voc_file):
    id_vec=id_to_vec(emb_file)
    return word_to_id(voc_file),id_vec,get_word_vec(id_vec)

class Data_data(object):
    def __init__(self, questions,questions2=None):
        super(Data_data, self).__init__()
        if questions2==None:
            self.A,self.input_data,self.targets= questions[0],questions[1],questions[2]
        else:
            self.A = torch.cat((questions[0],questions2[0]))
            self.input_data = torch.cat((questions[1],questions2[1]))
            self.targets = torch.cat((questions[2],questions2[2]))
        self.corpus_length=len(self.targets)
        self.start=0
    def next_batch(self,batch_size):
        start=self.start
        end=(self.start+batch_size) if (self.start+batch_size)<=self.corpus_length else self.corpus_length
        self.start=(self.start+batch_size)
        if self.start<self.corpus_length:
            epoch_flag=False
        else:
            self.start=self.start%self.corpus_length
            epoch_flag=True
        return [trans_to_cuda(self.A[start:end]),trans_to_cuda(self.input_data[start:end]),trans_to_cuda(self.targets[start:end])],epoch_flag

    def all_data(self,index=None):
        if type(index)==type(None):
            return [trans_to_cuda(self.A),trans_to_cuda(self.input_data),trans_to_cuda(self.targets)]
        else:
            return [trans_to_cuda(self.A.index_select(0,index)),trans_to_cuda(self.input_data.index_select(0,index)),trans_to_cuda(self.targets.index_select(0,index))]            
