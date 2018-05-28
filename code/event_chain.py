#coding:utf8
#
# This is the PairLSTM baseline model in our paper.
# We made a lot of modifications to the model in wang et al. emnlp2017, because the performace of the exact model 
# described in their emnlp2017 paper is very poor.
# Though they said they released their code at https://github.com/wangzq870305/event_chain, the code there doesn't make any sense.

from gnn_with_args import *

class EventChain(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size,word_vec,num_layers=1,bidirectional=False):
        super(EventChain, self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.num_directions= 1 if self.bidirectional==False else 2
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data = torch.from_numpy(word_vec)
        # self.embedding.weight.requires_grad=False
        self.gru = nn.GRU(self.embedding_dim, self.hidden_dim,self.num_layers,dropout=DROPOUT,bidirectional=self.bidirectional)
        self.linear_s_one=nn.Linear(hidden_dim*self.num_directions, 1,bias=False)
        self.linear_s_two=nn.Linear(hidden_dim*self.num_directions, 1,bias=True)
        self.linear_u_one=nn.Linear(hidden_dim*self.num_directions, 1,bias=False)
        self.linear_u_two=nn.Linear(hidden_dim*self.num_directions, 1,bias=True)
        self.loss_function = nn.MultiMarginLoss(margin=MARGIN)

        model_grad_params=filter(lambda p:p.requires_grad==True,self.parameters())
        train_params = list(map(id, self.embedding.parameters()))
        tune_params = filter(lambda p:id(p) not in train_params, model_grad_params)
        self.optimizer = optim.RMSprop([{'params':tune_params},{'params':self.embedding.parameters(),'lr':LR*0.06}],lr=LR, weight_decay=L2_penalty,momentum=0.2)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10,60], gamma=0.5)
        
    def compute_scores(self,output):  
        output=output.transpose(0, 1)   #5000*9*(128*4)
        a=self.linear_s_one(output[:,0:8,:])  
        b=self.linear_s_two(output[:,8,:])   
        c=torch.add(a.view(-1,8),b)  
        scores=F.sigmoid(c) 
        # attention weight matrix
        u_a=self.linear_u_one(output[:,0:8,:])
        u_b=self.linear_u_two(output[:,8,:])
        u_c=torch.add(u_a.view(-1,8),u_b)
        weight=torch.exp(F.tanh(u_c))
        weight=weight/torch.sum(weight,1).view(-1,1)
        scores=torch.sum(torch.mul(scores,weight),1).view(-1,5)
        # print (scores)
        return scores

    def forward(self, input):
        hidden = self.embedding(input) #1000*(13*4)*128
        hidden=torch.cat((hidden[:,0:13,:],hidden[:,13:26,:],hidden[:,26:39,:],hidden[:,39:52,:]),2) #1000*13*(128*4)
        input_a=hidden[:,0:8,:].repeat(1,5,1).view(5*len(hidden),8,-1) 
        input_b=hidden[:,8:13,:].contiguous().view(-1,1,512) 
        hidden=torch.cat((input_a,input_b),1) #5000*9*(128*4)

        self.hidden=self.init_hidden(len(hidden))
        output = hidden.transpose(0, 1) #9*5000*(128*4)
        output, self.hidden = self.gru(output, self.hidden)
        scores=self.compute_scores(output)
        return scores

    def predict(self, input, targets):
        scores=self.forward(input)
        sorted, L = torch.sort(scores,descending=True)
        num_correct0 = torch.sum((L[:,0] == targets).type(torch.FloatTensor))
        num_correct1 = torch.sum((L[:,1] == targets).type(torch.FloatTensor))
        num_correct2 = torch.sum((L[:,2] == targets).type(torch.FloatTensor))
        num_correct3 = torch.sum((L[:,3] == targets).type(torch.FloatTensor))
        num_correct4 = torch.sum((L[:,4] == targets).type(torch.FloatTensor))
        samples = len(targets)
        accuracy0 = num_correct0 / samples *100.0 
        accuracy1 = num_correct1 / samples *100.0 
        accuracy2 = num_correct2 / samples *100.0 
        accuracy3 = num_correct3 / samples *100.0 
        accuracy4 = num_correct4 / samples *100.0 
        return accuracy0,accuracy1,accuracy2,accuracy3,accuracy4

    def predict_with_minibatch(self,input,targets):
        scores=Variable(torch.zeros(len(targets),5)).cuda()
        for i in range(int(len(targets)/BATCH_SIZE)):
            scores_temp=self.forward(input[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
            scores[i*BATCH_SIZE:(i+1)*BATCH_SIZE]=scores_temp
        sorted, L = torch.sort(scores,descending=True)
        num_correct0 = torch.sum((L[:,0] == targets).type(torch.FloatTensor))
        num_correct1 = torch.sum((L[:,1] == targets).type(torch.FloatTensor))
        num_correct2 = torch.sum((L[:,2] == targets).type(torch.FloatTensor))
        num_correct3 = torch.sum((L[:,3] == targets).type(torch.FloatTensor))
        num_correct4 = torch.sum((L[:,4] == targets).type(torch.FloatTensor))
        samples = len(targets)
        accuracy0 = num_correct0 / samples *100.0 
        accuracy1 = num_correct1 / samples *100.0 
        accuracy2 = num_correct2 / samples *100.0 
        accuracy3 = num_correct3 / samples *100.0 
        accuracy4 = num_correct4 / samples *100.0 
        return accuracy0,accuracy1,accuracy2,accuracy3,accuracy4,scores

    def init_hidden(self,size):
        hidden = Variable(torch.zeros(self.num_layers * self.num_directions, size , self.hidden_dim))
        return trans_to_cuda(hidden)

    def weights_init(self,m):
        if isinstance(m, nn.GRU):
            nn.init.orthogonal(m.weight_hh_l0)
            nn.init.orthogonal(m.weight_ih_l0)
            nn.init.constant(m.bias_hh_l0,0)
            nn.init.constant(m.bias_ih_l0,0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            # nn.init.constant(m.bias,0)


def train():
    model=trans_to_cuda(EventChain(embedding_dim=HIDDEN_DIM,hidden_dim=HIDDEN_DIM,vocab_size=len(word_vec),word_vec=word_vec,num_layers=1,bidirectional=False))
    # model.apply(model.weights_init())
    acc_list=[]
    best_acc=0.0
    best_epoch=0
    print ('start training')
    EPO=0
    start=time.time()
    while True:
        patient=0
        for epoch in range(EPOCHES):
            model.optimizer.zero_grad() 
            data,epoch_flag=train_data.next_batch(BATCH_SIZE)
            # if epoch_flag:
            #     model.scheduler.step()    
            scores=model(data[1]) 
            loss = model.loss_function(scores,data[2])
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(),1)
            model.optimizer.step()
            data=dev_data.all_data()
            accuracy,accuracy1,accuracy2,accuracy3,accuracy4,scores2=model.predict_with_minibatch(data[1],data[2])
            # if (EPOCHES*EPO+epoch) % 50==0:
            print ('Epoch %d : Eval  Acc: %f, %f, %f, %f, %f' % (EPOCHES*EPO+epoch,accuracy.data[0],accuracy1.data[0],accuracy2.data[0],accuracy3.data[0],accuracy4.data[0]))
            acc_list.append((time.time()-start,accuracy.data[0]))
            if best_acc<accuracy.data[0]:
                best_acc=accuracy.data[0]
                if best_acc>=51:
                    torch.save(model.state_dict(), ('../data/event_chain_acc_%s_.model' % (best_acc)))
                best_epoch=EPOCHES*EPO+epoch+1
                patient=0
            else:
                patient+=1
            if patient>PATIENTS:
                break
        if epoch==(EPOCHES-1):
            EPO+=1
            continue
        else:
            break
    print ('Epoch %d : Best Acc: %f' % (best_epoch,best_acc))
    pickle.dump(acc_list,open('../data/lstm_acc_list.pickle','wb'),2)
    return best_acc,best_epoch

HIDDEN_DIM = 128*4
L2_penalty=1e-8
LR=0.0001
MARGIN=0.015
BATCH_SIZE=1000
EPOCHES=520
PATIENTS=500
DROPOUT=0.2

if __name__ == '__main__':
    dev_data=Data_data(pickle.load(open('../data/corpus_index_dev_with_args_all.data','rb')))
    test_data=Data_data(pickle.load(open('../data/corpus_index_test_with_args_all.data','rb')))
    train_data=Data_data(pickle.load(open('../data/corpus_index_train0_with_args_all.data','rb')))
    print('train data prepare done')
    word_id,id_vec,word_vec=get_hash_for_word('/users3/zyli/github/OpenNE/output/verb_net/1_property/deepwalk_128_unweighted_with_args.txt',verb_net3_mapping_with_args)
    print('word vector prepare done')
    start=time.time()
    best_acc,best_epoch=train()
    end=time.time()
    print ("Run time: %f s" % (end-start))
    with open('best_result.txt','a') as f:
        f.write('Best Acc: %f, Epoch %d , L2_penalty=%s ,MARGIN=%s ,LR=%s ,BATCH_SIZE=%s ,EPOCHES=%s ,PATIENTS=%s, HIDDEN_DIM=%s event-chain\n' % (best_acc,best_epoch,L2_penalty,MARGIN,LR,BATCH_SIZE,EPOCHES,PATIENTS,HIDDEN_DIM))
    f.close()
