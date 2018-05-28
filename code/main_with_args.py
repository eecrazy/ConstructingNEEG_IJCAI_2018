#coding:utf8
# Run this code to train our SGNN model.
# Generally we can train a model in about 1400 seconds (the code will automatically terminate by using early stop) using one Tesla P100 GPU.
from gnn_with_args import *

def main():
    dev_data=Data_data(pickle.load(open('../data/corpus_index_dev_with_args_all_chain.data','rb')))
    test_data=Data_data(pickle.load(open('../data/corpus_index_test_with_args_all_chain.data','rb')))
    train_data=Data_data(pickle.load(open('../data/corpus_index_train0_with_args_all_chain.data','rb')))
    ans=pickle.load(open('../data/dev.answer','rb'))
    dev_index=pickle.load(open('../data/dev_index.pickle','rb'))
    print('train data prepare done')
    word_id,id_vec,word_vec=get_hash_for_word('../data/deepwalk_128_unweighted_with_args.txt',verb_net3_mapping_with_args)
    print('word vector prepare done')

    if len(sys.argv)==9:
        L2_penalty,MARGIN,LR,T,BATCH_SIZE,EPOCHES,PATIENTS,METRIC=sys.argv[1:]
    else:
        HIDDEN_DIM = 128*4
        L2_penalty=0.00001
        LR=0.0001
        T=2
        MARGIN=0.015
        BATCH_SIZE=1000
        EPOCHES=520
        PATIENTS=500
        METRIC='euclid'

        if METRIC=='euclid':  #   
            L2_penalty=0.00001
            LR=0.0001
            BATCH_SIZE=1000
            MARGIN=0.015
            PATIENTS=500
        if METRIC=='dot':  # 
            # LR=0.004
            MARGIN=0.5
        if METRIC=='cosine': # 
            # LR=0.001
            MARGIN=0.05
        if METRIC=='norm_euclid': # 
            # LR=0.0011
            MARGIN=0.07
        if METRIC=='manhattan': # 
            # LR=0.0015
            MARGIN=4.5
        if METRIC=='multi': # 
            # LR=0.001
            MARGIN=0.015
        if METRIC=='nonlinear': # 
            # LR=0.001
            MARGIN=0.015
    start=time.time()
    best_acc,best_epoch=train(dev_index,word_vec,ans,train_data,dev_data,test_data,float(L2_penalty),float(MARGIN),float(LR),int(T),int(BATCH_SIZE),int(EPOCHES),int(PATIENTS),int(HIDDEN_DIM),METRIC)
    end=time.time()
    print ("Run time: %f s" % (end-start))
    with open('best_result.txt','a') as f:
        f.write('Best Acc: %f, Epoch %d , L2_penalty=%s ,MARGIN=%s ,LR=%s ,T=%s ,BATCH_SIZE=%s ,EPOCHES=%s ,PATIENTS=%s, HIDDEN_DIM=%s, METRIC=%s\n' % (best_acc,best_epoch,L2_penalty,MARGIN,LR,T,BATCH_SIZE,EPOCHES,PATIENTS,HIDDEN_DIM,METRIC))
    f.close()


if __name__ == '__main__':
    main()

# 事件表示：事件链条的多维分布表示，加入频率和共现频次信息
# 构建Graph: 统计bigram-过滤低频,删除自环,高频事件处理-图构建-计算概率
# Context Extension By Ranking
# Highway Networks
# SRU
# Attention
# Subgraph Embedding
# Adam
