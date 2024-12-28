import math
import logging
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import tqdm
from torch.nn.utils import clip_grad_norm_
import os

from EduKTM import KTM
from FRKTNet import FRKTNet

#device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

dataset_dir ={}
dataset_dir["assistment2009"] = "2009_skill_builder_data_corrected_collapsed"
dataset_dir["assistment2012"] = "2012-2013-data-with-predictions-4-final"
dataset_dir["assistment2015"] = "2015_100_skill_builders_main_problems"
dataset_dir["assistment2017"] = "anonymized_full_release_competition_dataset"
dataset_dir["algebra2005"] = "algebra_2005_2006"
dataset_dir["statics"] = "statics"
dataset_dir["EdNet-KT1"] = "EdNet-Contents/contents"



def binary_entropy(target, pred):
    loss = target * np.log(np.maximum(1e-10, pred)) + (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * -1.0


def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train_one_epoch(net, optimizer, criterion, batch_size, device, s_data, a_data, e_data, at_data):
    net.train()
    n = int(math.ceil(len(a_data) / batch_size))
    
    shuffled_ind = np.arange(a_data.shape[0])
    np.random.shuffle(shuffled_ind)
    s_data = s_data[shuffled_ind]
    a_data = a_data[shuffled_ind]
    e_data = e_data[shuffled_ind]
    at_data = at_data[shuffled_ind]

    pred_list = []
    target_list = []

    for idx in tqdm.tqdm(range(n), 'Training'):
        optimizer.zero_grad()

        s_one_seq = s_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        
        input_s = torch.from_numpy(s_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)
        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)

        pred = net(input_s, target, input_e, input_at)

        mask = input_s[:, 1:] > 0
        masked_pred = pred[:, 1:][mask]
        masked_truth = target[:, 1:][mask]

        loss = criterion(masked_pred, masked_truth).sum()

        loss.backward()
        clip_grad_norm_(net.parameters(), max_norm=1.0)   #seq_len<=200
        #clip_grad_norm_(net.parameters(), max_norm=0.1)
        #print("know_params.grad :",net.know_params.grad)
        optimizer.step()

        masked_pred = masked_pred.detach().cpu().numpy()
        masked_truth = masked_truth.detach().cpu().numpy()

        pred_list.append(masked_pred)
        target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


def test_one_epoch(net, batch_size, device, s_data, a_data, e_data, at_data):
    net.eval()
    n = int(math.ceil(len(e_data) / batch_size))

    pred_list = []
    target_list = []

    for idx in tqdm.tqdm(range(n), 'Testing'):
        s_one_seq = s_data[idx * batch_size: (idx + 1) * batch_size, :]
        a_one_seq = a_data[idx * batch_size: (idx + 1) * batch_size, :]
        e_one_seq = e_data[idx * batch_size: (idx + 1) * batch_size, :]
        at_one_seq = at_data[idx * batch_size: (idx + 1) * batch_size, :]
        
        input_s = torch.from_numpy(s_one_seq).long().to(device)
        target = torch.from_numpy(a_one_seq).float().to(device)
        input_e = torch.from_numpy(e_one_seq).long().to(device)
        input_at = torch.from_numpy(at_one_seq).long().to(device)

        with torch.no_grad():
            pred = net(input_s, target, input_e, input_at)

            mask = input_e[:, 1:] > 0
            masked_pred = pred[:, 1:][mask].detach().cpu().numpy()
            masked_truth = target[:, 1:][mask].detach().cpu().numpy()

            pred_list.append(masked_pred)
            target_list.append(masked_truth)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binary_entropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, auc, accuracy


    

from load_data import DATA
from load_data_akt import DATA as DT_DATA             # 来自AKT的数据集
from load_data_akt import PID_DATA as DT_PID_DATA


class FRKT(KTM):
    def __init__(self, n_exercise, n_skill, batch_size ,q_matrix, device, hidden_size=128, k_components = 32, dropout=0.2, use_rasch = True):
        super(FRKT, self).__init__()
        q_matrix = torch.from_numpy(q_matrix).float().to(device)
        self.frkt_net = FRKTNet(n_exercise, n_skill, q_matrix, device, hidden_size,k_components, dropout, use_rasch).to(device)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.k_components = k_components
        self.n_skill = n_skill 
        self.n_exercise = n_exercise
        self.device = device
        self.use_rasch = 1 if use_rasch else 0

    def train(self, dataset_name, *, seq_len = 200, epoch: int, lr=0.001, lr_decay_step=15, lr_decay_rate=0.5) -> ...:
        optimizer = torch.optim.Adam(self.frkt_net.parameters(), lr=lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.0005)
        #optimizer = torch.optim.SGD(self.frkt_net.parameters(), lr=lr,momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_step, gamma=lr_decay_rate)
        criterion = nn.BCELoss(reduction='none')
        
        data_dir = "../../data/"+dataset_dir[dataset_name]
        if dataset_name in  dataset_name in ["assistment2009", "assistment2012", "assistment2015", "assistment2017","EdNet-KT1","algebra2005"]:
            train_path = os.path.join(data_dir, "train0.txt")
            valid_path = os.path.join(data_dir, "valid0.txt")
            
            dat = DATA(seqlen=seq_len, separate_char=',')
        elif dataset_name in ["statics"]:   # AKT Dataset
            train_path = os.path.join(data_dir, "statics_train1.csv")
            valid_path = os.path.join(data_dir, "statics_valid1.csv")
    
            if self.n_exercise>0:
                dat = DT_PID_DATA(seqlen=seq_len, separate_char=',')
            else:
                dat = DT_DATA(seqlen=seq_len, separate_char=',')
        else:
            raise ValueError('ValueError: Unknown dataset! ')
        print(train_path)
        train_data = dat.load_data(train_path)
        valid_data = dat.load_data(valid_path)
            
        best_train_auc, best_train_accuracy = .0, .0
        best_valid_auc, best_valid_accuracy = .0, .0
        early_stop = 5
        count = 0
        
        saved_path = os.path.join("saved_model",dataset_dir[dataset_name])
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
            
        for idx in range(epoch):
            if count>early_stop:
                break
            train_loss, train_auc, train_accuracy = train_one_epoch(self.frkt_net, optimizer, criterion,
                                                                    self.batch_size, self.device, *train_data)
            print("[Epoch %d] LogisticLoss: %.6f, auc: %.6f, accuracy: %.6f " % (idx, train_loss, train_auc, train_accuracy))
            if train_auc > best_train_auc:
                best_train_auc = train_auc
                best_train_accuracy = train_accuracy

           # scheduler.step()
         
            if valid_data is not None:
                valid_loss, valid_auc, valid_accuracy = self.eval(self.device, valid_data)
                print("[Epoch %d] LogisticLoss: %.6f, auc: %.6f, accuracy: %.6f" % (idx, valid_loss, valid_auc, valid_accuracy))
                if valid_auc > best_valid_auc:
                    best_valid_auc = valid_auc
                    best_valid_accuracy = valid_accuracy
                    count=0
                    
                    model_path = os.path.join(saved_path, f"model-seq_len{seq_len:03d}-lr{lr}-hidden_size{self.hidden_size:03d}-k{self.k_components}-use_rasch{self.use_rasch:01d}.pt")
                    self.save(model_path)
                else:
                    count+=1
                    
        return best_train_auc, best_train_accuracy, best_valid_auc, best_valid_accuracy

    def eval(self, device, test_data) -> ...:
        self.frkt_net.eval()
        return test_one_epoch(self.frkt_net, self.batch_size, device, *test_data)

    def save(self, filepath) -> ...:
        torch.save(self.frkt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath) -> ...:
        self.frkt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
        

    # **********************将训练集减半*********************************
    
    # 1.只使用前一半
    # cliped_len = len(e_data)//2
    # e_data = e_data[:cliped_len]
    # at_data = at_data[:cliped_len]
    # a_data = a_data[:cliped_len]
    # n = int(math.ceil(cliped_len / batch_size))
    
    # shuffled_ind = np.arange(cliped_len)
    # np.random.shuffle(shuffled_ind)
    # e_data = e_data[shuffled_ind]
    # at_data = at_data[shuffled_ind]
    # a_data = a_data[shuffled_ind]
    
     # 2.随机抽一半
    # total_len = len(e_data)
    # cliped_len = len(e_data)//2
    # sampled_indices = np.random.choice(total_len, size=cliped_len, replace=False)
    # e_data = e_data[sampled_indices]
    # at_data = at_data[sampled_indices]
    # a_data = a_data[sampled_indices]
    # n = int(math.ceil(cliped_len / batch_size))
    
    # **********************将训练集减半*********************************