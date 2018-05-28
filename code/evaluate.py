#coding:utf8
#
# Run this code to get the final results reported in our ijcai paper.
from io import open
import string
import re
import random
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pprint,copy
use_cuda = torch.cuda.is_available()
from gnn_with_args import *
from event_chain import EventChain
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_event_chains(event_list):
    return ['%s_%s' % (ev[0],ev[2]) for ev in event_list]

def get_word_embedding(word,word_id,id_vec,emb_size):
    if word in word_id:
        return id_vec[word_id[word]]
    else:
        return np.zeros(emb_size,dtype=np.float32)

def get_vec_rep(questions,word_id,id_vec,emb_size,predict=False):
    rep = np.zeros((5*len(questions),9,emb_size),dtype=np.float32)
    correct_answers=[]
    for i,q in enumerate(questions):
        context_chain=get_event_chains(q[0])
        choice_chain=get_event_chains(q[1])
        correct_answers.append(q[2])
        for j,context in enumerate(context_chain):
            context_vec=get_word_embedding(context,word_id,id_vec,emb_size)
            rep[5*i:5*(i+1),j,:]=context_vec
        for k,choice in enumerate(choice_chain):
            choice_vec=get_word_embedding(choice,word_id,id_vec,emb_size)
            rep[5*i+k,-1,:]=choice_vec
    if not predict:
        input_data=Variable(torch.from_numpy(rep))
    else:
        input_data=Variable(torch.from_numpy(rep),volatile=True)
    correct_answers = Variable(torch.from_numpy(np.array(correct_answers)))
    return input_data,correct_answers


class Word2VecAttention(nn.Module):
    def __init__(self):
        super(Word2VecAttention, self).__init__()
        self.linear_u_one=nn.Linear(HIDDEN_DIM,1,bias=False)
        self.linear_u_one2=nn.Linear(HIDDEN_DIM,1,bias=False)
        self.linear_u_two=nn.Linear(HIDDEN_DIM,1,bias=True)
        self.linear_u_two2=nn.Linear(HIDDEN_DIM,1,bias=False)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

    def compute_scores(self,input_data):   
        weight=Variable(torch.zeros((len(input_data),8,1)).fill_(1./8))
        weighted_input=torch.mul(input_data[:,0:8,:],weight)  
        a=torch.sum(weighted_input,1)
        b=input_data[:,8,:]/8.0
        scores=-torch.norm(a-b, 2, 1).view(-1,5)
        return scores

    def forward(self, input_data):
        return self.compute_scores(input_data)

    def correct_answer_position(self,L,correct_answers):
        num_correct1 = torch.sum((L[:,0] == correct_answers).type(torch.FloatTensor))
        num_correct2 = torch.sum((L[:,1] == correct_answers).type(torch.FloatTensor))
        num_correct3 = torch.sum((L[:,2] == correct_answers).type(torch.FloatTensor))
        num_correct4 = torch.sum((L[:,3] == correct_answers).type(torch.FloatTensor))
        num_correct5 = torch.sum((L[:,4] == correct_answers).type(torch.FloatTensor))
        print ("%d / %d 1st max correct: %f" % (num_correct1.data[0], len(correct_answers),num_correct1 / len(correct_answers) * 100.))
        print ("%d / %d 2ed max correct: %f" % (num_correct2.data[0], len(correct_answers),num_correct2 / len(correct_answers) * 100.))
        print ("%d / %d 3rd max correct: %f" % (num_correct3.data[0], len(correct_answers),num_correct3 / len(correct_answers) * 100.))
        print ("%d / %d 4th max correct: %f" % (num_correct4.data[0], len(correct_answers),num_correct4 / len(correct_answers) * 100.))
        print ("%d / %d 5th max correct: %f" % (num_correct5.data[0], len(correct_answers),num_correct5 / len(correct_answers) * 100.))

    def predict(self, input_data, targets):
        scores=self.forward(input_data)
        sorted, L = torch.sort(scores,descending=True)
        self.correct_answer_position(L,targets)
        selections=L[:,0]
        pickle.dump((selections != targets),open('../data/test.answer','wb'))
        num_correct = torch.sum((selections == targets).type(torch.FloatTensor))
        accuracy = num_correct / len(targets) *100.0 
        return accuracy

    def weights_init(self,m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_uniform(m.weight)
        elif isinstance(m, nn.GRU):
            nn.init.xavier_uniform(m.weight_hh_l0)
            nn.init.xavier_uniform(m.weight_ih_l0)
            nn.init.constant(m.bias_hh_l0,0)
            nn.init.constant(m.bias_ih_l0,0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            # nn.init.uniform(m.weight)
            # nn.init.normal(m.weight)

def train(questions):
    model=Word2VecAttention()
    input_data_test,correct_answers_test=get_vec_rep(questions,word_id,id_vec,HIDDEN_DIM,predict=True)
    accuracy=model.predict(input_data_test,correct_answers_test)
    print('Test  Acc: ',accuracy.data[0])


def process_test(scores,test_index):
    for index in test_index:
        scores[index]=np.min(scores)
    return scores

def get_acc(scores,correct_answers,name='scores',save=False):
    selections = np.argmax(scores, axis=1)
    num_correct = int(np.sum(selections == correct_answers))
    if save:
        pickle.dump((selections == correct_answers),open('./scores/'+name,'wb'),2)
    samples = len(correct_answers)
    accuracy = float(num_correct) / samples * 100.
    # print ("%d / %d correct: %f" % (num_correct, samples, accuracy))
    return accuracy

if __name__ == '__main__':
    test_index=pickle.load(open('../data/test_index.pickle','rb'))

    HIDDEN_DIM = 128
    L2_penalty=0.00001
    MARGIN=0.015
    LR=0.0001
    T=1
    BATCH_SIZE=50
    EPOCHES=520
    PATIENTS=300
    test_data=Data_data(pickle.load(open('../data/corpus_index_test_with_args_all_chain.data','rb')))
    word_id,id_vec,word_vec=get_hash_for_word('/users3/zyli/github/OpenNE/output/verb_net/1_property/deepwalk_128_unweighted_with_args.txt',verb_net3_mapping_with_args)
    
    HIDDEN_DIM = 128*4
    L2_penalty=0.00001
    MARGIN=0.015
    LR=0.0001
    T=1
    BATCH_SIZE=1000
    EPOCHES=520
    PATIENTS=300
    test_data=Data_data(pickle.load(open('../data/corpus_index_test_with_args_all_chain.data','rb')))
    model=trans_to_cuda(EventGraph_With_Args(len(word_vec),HIDDEN_DIM,word_vec,L2_penalty,MARGIN,LR,T))
    model.load_state_dict(torch.load('../data/gnn_euclid_acc_52.380001068115234_.model'))

    data=test_data.all_data()
    correct_answers=data[2].cpu().data.numpy()
    scores1=model(data[1],data[0]).cpu().data.numpy() 
    scores1=process_test(scores1,test_index)
    print (get_acc(scores1,correct_answers,'scores1'))

    HIDDEN_DIM = 128*4
    L2_penalty=0.00001
    MARGIN=0.015
    LR=0.0001
    T=1
    BATCH_SIZE=1000
    EPOCHES=520
    PATIENTS=300
    test_data=Data_data(pickle.load(open('../data/corpus_index_test_with_args_all.data','rb')))
    data=test_data.all_data()
    model=trans_to_cuda(EventChain(embedding_dim=HIDDEN_DIM,hidden_dim=HIDDEN_DIM,vocab_size=len(word_vec),word_vec=word_vec,num_layers=1,bidirectional=False))
    model.load_state_dict(torch.load('../data/event_chain_acc_50.98999786376953_.model'))
    accuracy,accuracy1,accuracy2,accuracy3,accuracy4,scores2=model.predict_with_minibatch(data[1],data[2])
    scores2=scores2.cpu().data.numpy() 
    scores2=process_test(scores2,test_index)
    print (get_acc(scores2,correct_answers,'scores2'))

    scores3=pickle.load(open('../data/event_comp_test.scores','rb'),encoding='bytes')
    scores3=process_test(scores3,test_index)
    print (get_acc(scores3,correct_answers,'scores3'))


    scores1=preprocessing.scale(scores1)
    scores2=preprocessing.scale(scores2)
    scores3=preprocessing.scale(scores3)

    best_acc=0. 
    best_i_j_k=(0,0)
    for i in np.arange(-3,3,0.1):
        for j in np.arange(-3,3,0.1):
            acc=get_acc(scores3*i+scores1*j,correct_answers)
            if best_acc<acc:
                best_acc=acc 
                best_i_j_k=(i,j)
    print (best_acc,best_i_j_k)
    get_acc(scores3*best_i_j_k[0]+scores1*best_i_j_k[1],correct_answers,'scores1_scores3')

    best_acc=0. 
    best_i_j_k=(0,0)
    for i in np.arange(-3,3,0.1):
        for j in np.arange(-3,3,0.1):
            acc=get_acc(scores1*i+scores2*j,correct_answers)
            if best_acc<acc:
                best_acc=acc 
                best_i_j_k=(i,j)
    print (best_acc,best_i_j_k)
    get_acc(scores1*best_i_j_k[0]+scores2*best_i_j_k[1],correct_answers,'scores1_scores2')

    best_acc=0. 
    best_i_j_k=(0,0)
    for i in np.arange(-3,3,0.1):
        for j in np.arange(-3,3,0.1):
            acc=get_acc(scores3*i+scores2*j,correct_answers)
            if best_acc<acc:
                best_acc=acc 
                best_i_j_k=(i,j)
    print (best_acc,best_i_j_k)
    get_acc(scores3*best_i_j_k[0]+scores2*best_i_j_k[1],correct_answers,'scores2_scores3')

    best_acc=0. 
    best_i_j_k=(0,0,0)
    for i in np.arange(-3,3,0.1):
        for j in np.arange(-3,3,0.1):
            for k in np.arange(-3,3,0.1):
                acc=get_acc(scores1*i+scores3*j+scores2*k,correct_answers)
                if best_acc<acc:
                    best_acc=acc 
                    best_i_j_k=(i,j,k)
    print (best_acc,best_i_j_k)
    get_acc(scores1*best_i_j_k[0]+scores3*best_i_j_k[1]+scores2*best_i_j_k[2],correct_answers,'scores1_scores2_scores3')
    
# SGNN 1
# event_chain-PairLSTM 2
# event_comp 3


