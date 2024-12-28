import torch            
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_,constant_
import math

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        

class FRKTNet(nn.Module):
    def __init__(self, n_exercise, n_skill, q_matrix, device, hidden_size =128, k_components=32, dropout=0.2, use_rasch = True):
        super(FRKTNet, self).__init__()
        if  use_rasch and n_exercise>0:   # means use q_matrix to get mean of skill embedding 
            if not q_matrix.any():
                print("Illegal input")
                raise ValueError('ValueError:when use_rasch,the q_matrix can not be None ! ')
        self.n_exercise = n_exercise
        self.n_skill = n_skill
        self.hidden_size = hidden_size
        self.k_components = k_components
        self.use_rasch = use_rasch
        self.q_matrix = q_matrix
        self.device = device
        
        # user for rachel embedding *********
        
        self.s_embed = nn.Embedding(n_skill + 1, hidden_size)
        torch.nn.init.xavier_uniform_(self.s_embed.weight)
        self.a_embed = nn.Embedding(2, hidden_size)
        torch.nn.init.xavier_uniform_(self.a_embed.weight)
        if n_exercise > 0:
            # just use the exercise lable
            self.e_embed = nn.Embedding(n_exercise+ 1, hidden_size)      # load_data , n_exercise =0 时，问题标签同概念标签(assist2015,statics（kddcup）仅有概念标签)
            torch.nn.init.xavier_uniform_(self.e_embed.weight)  
            self.linear_1 = nn.Linear(2* hidden_size, hidden_size)                 # e + s
            torch.nn.init.xavier_uniform_(self.linear_1.weight)
            self.linear_2 = nn.Linear(3* hidden_size, hidden_size)                 # e + s + a
            torch.nn.init.xavier_uniform_(self.linear_2.weight)
            self.linear_3 = nn.Linear(2* hidden_size, hidden_size)                 # e + a
            torch.nn.init.xavier_uniform_(self.linear_3.weight)
        else:
            self.linear_1 = nn.Linear(2* hidden_size, hidden_size)                 # s + a
            torch.nn.init.xavier_uniform_(self.linear_1.weight)
            
        self.key_matrix = nn.Parameter(torch.zeros(self.k_components, hidden_size))   # +1 OR NOT
        #torch.nn.init.uniform_(self.key_matrix, -1.0, 1.0)
        torch.nn.init.xavier_uniform_(self.key_matrix) 
        
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers =1,batch_first = True)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

        self.linear_gain = nn.Linear(2* hidden_size, hidden_size)                
        torch.nn.init.xavier_uniform_(self.linear_gain.weight)
        self.linear_gain_gate = nn.Linear(2* hidden_size, hidden_size)                
        torch.nn.init.xavier_uniform_(self.linear_gain_gate.weight)
        self.linear_forget_gate = nn.Linear(2* hidden_size, hidden_size)                
        torch.nn.init.xavier_uniform_(self.linear_forget_gate.weight)
        self.predict_fc = nn.Linear(3*hidden_size,1)                            # 预测  input_dim = 2*know_params.shape(0)
        torch.nn.init.xavier_uniform_(self.predict_fc.weight)
        self.predict_fc1 = nn.Linear(3*hidden_size,1)                            # 预测  input_dim = 2*know_params.shape(0)
        torch.nn.init.xavier_uniform_(self.predict_fc1.weight)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
          
    def embedding(self, s_data, a_data, e_data):     # use rasch model to combine the e2s information   
        batch_size, seq_len = a_data.size(0), a_data.size(1)
        if self.n_exercise>0:
            a_data = a_data.long()   # 传入的是float，改为long才能作为Embedding 的索引
            related_skills = self.q_matrix[e_data]  # b len n_skill+1
            sum_skills = torch.sum(related_skills,dim=-1,keepdim=True) #是否使用平均待定
            sum_skills = torch.where(sum_skills == 0, 1, sum_skills)
            s_emb_data = torch.matmul(related_skills,self.s_embed.weight[None,:,:].expand(batch_size,-1,-1))
            s_emb_data = s_emb_data/sum_skills
            e_emb_data = self.e_embed(e_data)
            #se_data = s_emb_data + e_emb_data
            se_data = torch.relu(self.linear_1(torch.cat([s_emb_data, e_emb_data],dim=-1)))
            a_emb_data = self.a_embed(a_data)
            #se_learning = se_data + a_emb_data
            se_learning = torch.relu(self.linear_2(torch.cat([s_emb_data, e_emb_data, a_emb_data],dim=-1)))
            #e_learning = e_emb_data+a_emb_data
            e_learning = torch.relu(self.linear_3(torch.cat([e_emb_data,a_emb_data],dim=-1)))
            return se_data, se_learning, e_emb_data, e_learning
        else:
            a_data = a_data.long()   # 传入的是float，改为long才能作为Embedding 的索引
            s_emb_data = self.s_embed(s_data)
            a_emb_data = self.a_embed(a_data)
            #s_learning = s_emb_data + a_emb_data
            s_learning = torch.relu(self.linear_1(torch.cat([s_emb_data, a_emb_data],dim=-1)))
            return s_emb_data, s_learning, s_emb_data, s_learning
            
        
    def forward(self, s_data, a_data, e_data, at_data):
        batch_size, seq_len = a_data.size(0), a_data.size(1)
        se_data, se_learning, e_emb_data, e_learning = self.embedding(s_data, a_data, e_data)
        
        h_0 = nn.init.xavier_uniform_(torch.zeros(1, self.hidden_size)).repeat(batch_size, 1).to(self.device)  # single state 
        out, state = self.gru(e_learning,h_0.unsqueeze(0))
        out = out[:,:-1,:]
        out = self.dropout(out)
        
        key_matrix=self.key_matrix
        h_pre = nn.init.xavier_uniform_(torch.zeros(self.k_components, self.hidden_size)).repeat(batch_size, 1, 1).to(self.device)
        h_tilde_pre = None
        h_tilde_output = []
        for t in range(0, seq_len - 1):
            e = se_data[:, t]     # batch_size * hidden_size
            weight = e[:,None,:] @ key_matrix[None,:,:].expand(batch_size,-1,-1).transpose(-1,-2)                 # b 1 hidden_size @  b hidden_size 16    ->> b 1 16
            if h_tilde_pre is None:
                h_tilde_pre = weight.bmm(h_pre).view(batch_size, self.hidden_size)   
            learning = se_learning[:, t]
            learning_gain = self.linear_gain(torch.cat((h_tilde_pre, learning), 1))
            learning_gain = self.tanh(learning_gain)
            gain_gate = self.sig(self.linear_gain_gate(torch.cat((h_tilde_pre, learning), 1)))
            LG = gain_gate * learning_gain
            LG_tilde = weight.transpose(1, 2).bmm(LG.view(batch_size, 1, -1))    
            LG_tilde = self.dropout(LG_tilde)        
            gamma_f = self.sig(self.linear_forget_gate(torch.cat((
                h_pre,
                LG.repeat(1, self.k_components).view(batch_size, -1, self.hidden_size),
            ), 2)))
            h = gamma_f * h_pre + LG_tilde
            weight_tilde = se_data[:, t+1][:,None,:] @ key_matrix[None,:,:].expand(batch_size,-1,-1).transpose(-1,-2)     
            h_tilde =weight_tilde.bmm(h).view(batch_size, self.hidden_size)
            h_tilde_output.append(h_tilde)
            h_pre = h
            h_tilde_pre = h_tilde
        
        out1 = torch.cat([h.unsqueeze(1) for h in h_tilde_output],dim=1)
        e_emb_data= e_emb_data[:,1:,:]
        se_data = se_data[:,1:,:]
        res = self.sig(self.predict_fc(torch.cat([out, out1.detach(), e_emb_data],dim = -1))).squeeze()   # detach1
        res1 = self.sig(self.predict_fc1(torch.cat([out.detach(), out1, se_data],dim = -1))).squeeze()
        pred = torch.zeros(batch_size, seq_len).to(self.device)
        pred[:,1:] = res * res1
        return pred

