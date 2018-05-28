#coding:utf8
# This is the SGNN model described in our ijcai paper.
from utils import *

class FNN(Module):
    def __init__(self, hidden_size,dropout_p=0.2):
        super(FNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear_one=nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        self.linear_two=nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        self.reset_parameters()

    def forward(self,hidden): #1000*13*128
        hidden1=F.sigmoid(self.linear_one(hidden))
        hidden2=self.linear_two(hidden1)
        return hidden2+hidden

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

class GNN(Module):
    def __init__(self, hidden_size,T,dropout_p=0.2):
        super(GNN, self).__init__()
        self.hidden_size = hidden_size
        self.T = T
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))        
        self.b_ah = Parameter(torch.Tensor(self.hidden_size))
        
        self.w_ih_2 = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.w_hh_2 = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih_2 = Parameter(torch.Tensor(self.gate_size))
        self.b_hh_2 = Parameter(torch.Tensor(self.gate_size))        
        self.b_ah_2 = Parameter(torch.Tensor(self.hidden_size))

        # self.dropout=nn.Dropout(dropout_p)
        self.reset_parameters()

    def GNNCell(self, A, hidden, w_ih, w_hh, b_ih, b_hh, b_ah):
        input=torch.matmul(A.transpose(1,2),hidden)+b_ah
        # input=self.dropout(input)
        gi = F.linear(input, w_ih, b_ih)
        gh = F.linear(hidden, w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        # hy=self.dropout(hy)
        return hy

    def forward(self, A, hidden):
        hidden1=self.GNNCell(A,hidden,self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.b_ah)
        hidden2=self.GNNCell(A,hidden1,self.w_ih, self.w_hh, self.b_ih, self.b_hh, self.b_ah)
        return hidden2

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

class EventGraph_With_Args(Module):
    def __init__(self, vocab_size, hidden_dim,word_vec,L2_penalty,MARGIN,LR,T,BATCH_SIZE=1000,dropout_p=0.2):
        super(EventGraph_With_Args, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size=vocab_size
        self.batch_size=BATCH_SIZE
        self.embedding = nn.Embedding(self.vocab_size,self.hidden_dim)
        self.embedding.weight.data = torch.from_numpy(word_vec)
        # self.embedding.weight.requires_grad=False
        self.gnn = GNN(self.hidden_dim,T)
        # self.fnn = FNN(self.hidden_dim)
        
        # compute
        self.linear_s_one=nn.Linear(hidden_dim, 1,bias=False)
        self.linear_s_two=nn.Linear(hidden_dim, 1,bias=True)
        self.linear_u_one=nn.Linear(hidden_dim,int(0.5*hidden_dim),bias=True)
        self.linear_u_one2=nn.Linear(int(0.5*hidden_dim),1,bias=True)
        self.linear_u_two=nn.Linear(hidden_dim,int(0.5*hidden_dim),bias=True)
        self.linear_u_two2=nn.Linear(int(0.5*hidden_dim),1,bias=True)
        # end compute
        
        self.multi = Parameter(torch.ones(3))
        self.dropout=nn.Dropout(dropout_p)
        self.loss_function = nn.MultiMarginLoss(margin=MARGIN)

        model_grad_params=filter(lambda p:p.requires_grad==True,self.parameters())
        train_params = list(map(id, self.embedding.parameters()))
        tune_params = filter(lambda p:id(p) not in train_params, model_grad_params)
        
        self.optimizer = optim.RMSprop([{'params':tune_params},{'params':self.embedding.parameters(),'lr':LR*0.06}],lr=LR, weight_decay=L2_penalty,momentum=0.2)

        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.995)

    def compute_scores(self,hidden,metric='euclid'):   #batch_size*13*128
        # attention on input 
        input_a=hidden[:,0:8,:].repeat(1,5,1).view(5*len(hidden),8,-1) 
        input_b=hidden[:,8:13,:] 
        u_a=F.relu(self.linear_u_one(input_a)) 
        u_a2=F.relu(self.linear_u_one2(u_a)) 
        u_b=F.relu(self.linear_u_two(input_b)) 
        u_b2=F.relu(self.linear_u_two2(u_b)) 
        u_c=torch.add(u_a2.view(5*len(hidden),8),u_b2.view(5*len(hidden),1))
        weight=torch.exp(F.tanh(u_c))
        weight=(weight/torch.sum(weight,1).view(-1,1)).view(-1,8,1)
        # weight.fill_(1./8) 
        weighted_input=torch.mul(input_a,weight) 
        a=torch.sum(weighted_input,1)
        b=input_b/8.0
        b=b.view(5*len(hidden),-1)
        if metric=='dot':
            scores=self.metric_dot(a,b)
        elif metric=='cosine':
            scores=self.metric_cosine(a,b)
        elif metric=='euclid':
            scores=self.metric_euclid(a,b)
        elif metric=='norm_euclid':
            scores=self.metric_norm_euclid(a,b)
        elif metric=='manhattan':
            scores=self.metric_manhattan(a,b)
        elif metric=='multi':
            scores=self.multi[0]*self.metric_euclid(a,b)+self.multi[1]*self.metric_dot(a,b)+self.multi[2]*self.metric_cosine(a,b)
        return scores

    def forward(self, input,A,metric='euclid',nn_type='gnn'):
        hidden = self.embedding(input)  #batch_size*(13*4)*128
        hidden=torch.cat((hidden[:,0:13,:],hidden[:,13:26,:],hidden[:,26:39,:],hidden[:,39:52,:]),2)        
        if nn_type=='gnn':
            hidden = self.gnn(A,hidden)


        # elif nn_type=='fnn':
        # hidden = self.fnn(hidden)
        scores=self.compute_scores(hidden,metric)

        return scores

    def predict(self,input,A,targets,dev_index,metric='euclid'):
        scores=self.forward(input,A,metric)
        # input和scores处理一下
        for index in dev_index:
            scores[index]=-100.0
        # 处理完毕
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

    def metric_dot(self, v0, v1):
        return torch.sum(v0*v1,1).view(-1,5)

    def metric_cosine(self, v0, v1):
        return F.cosine_similarity(v0,v1).view(-1,5)

    def metric_euclid(self, v0, v1):
        return -torch.norm(v0-v1, 2, 1).view(-1,5)

    def metric_norm_euclid(self, v0, v1):
        v0 = v0/torch.norm(v0, 2, 1).view(-1,1)
        v1 = v1/torch.norm(v1, 2, 1).view(-1,1)
        return -torch.norm(v0-v1, 2, 1).view(-1,5)

    def metric_manhattan(self, v0, v1):
        return -torch.sum(torch.abs(v0 - v1), 1).view(-1,5)

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

    def predict_with_minibatch(self,input,A,targets,dev_index,metric='euclid'):
        # input.volatile=True
        scores=trans_to_cuda(Variable(torch.zeros(len(targets),5)))
        for i in range(int(len(targets)/self.batch_size)):
            scores[i*self.batch_size:(i+1)*self.batch_size]=self.forward(input[i*self.batch_size:(i+1)*self.batch_size],A[i*self.batch_size:(i+1)*self.batch_size],metric)

        for index in dev_index:
            scores[index]=-100.0
        sorted, L = torch.sort(scores,descending=True)
        # self.correct_answer_position(L,targets)
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

    def weights_init(self,m):
        if isinstance(m, nn.GRU):
            nn.init.xavier_uniform(m.weight_hh_l0)
            nn.init.xavier_uniform(m.weight_ih_l0)
            nn.init.constant(m.bias_hh_l0,0)
            nn.init.constant(m.bias_ih_l0,0)
        elif isinstance(m, GNN):
            nn.init.xavier_uniform(m.w_hh)
            nn.init.xavier_uniform(m.w_ih)
            nn.init.xavier_uniform(m.w_hh_2)
            nn.init.xavier_uniform(m.w_ih_2)
            nn.init.constant(m.b_hh,0)
            nn.init.constant(m.b_ih,0)
            nn.init.constant(m.b_ah,0)
            nn.init.constant(m.b_hh_2,0)
            nn.init.constant(m.b_ih_2,0)
            nn.init.constant(m.b_ah_2,0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)

def train(dev_index,word_vec,ans,train_data,dev_data,test_data,L2_penalty,MARGIN,LR,T,BATCH_SIZE,EPOCHES,PATIENTS,HIDDEN_DIM,METRIC='euclid'):
    model=trans_to_cuda(EventGraph_With_Args(len(word_vec),HIDDEN_DIM,word_vec,L2_penalty,MARGIN,LR,T,BATCH_SIZE))   
    model.optimizer.zero_grad() 
    # model.scheduler.step()
    # model.apply(model.weights_init)
    acc_list=[]
    best_acc=0.0
    best_epoch=0
    print ('start training')
    EPO=0
    start=time.time()
    while True:
        patient=0
        for epoch in range(EPOCHES):
            data,epoch_flag=train_data.next_batch(BATCH_SIZE)
            model.train()
            scores=model(data[1],data[0],metric=METRIC) 
            loss = model.loss_function(scores,data[2])
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            # if (EPOCHES*EPO+epoch+1) % (1000/BATCH_SIZE)==0:
            data=dev_data.all_data()
            model.eval()
            accuracy,accuracy1,accuracy2,accuracy3,accuracy4=model.predict(Variable(data[1].data,volatile=True),data[0],data[2],dev_index,metric=METRIC)
            if (EPOCHES*EPO+epoch) % 50==0:
                print ('Epoch %d : Eval  Acc: %f, %f, %f, %f, %f, %s' % (EPOCHES*EPO+epoch,accuracy.data[0],accuracy1.data[0],accuracy2.data[0],accuracy3.data[0],accuracy4.data[0],METRIC))
            acc_list.append((time.time()-start,accuracy.data[0]))
            if best_acc<accuracy.data[0]:
                best_acc=accuracy.data[0]
                if best_acc>=52.7:
                    torch.save(model.state_dict(), ('../data/gnn_%s_acc_%s_.model' % (METRIC,best_acc)))
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
    # pickle.dump(acc_list,open('../output/gnn_acc_list.pickle','wb'),2)
    return best_acc,best_epoch
