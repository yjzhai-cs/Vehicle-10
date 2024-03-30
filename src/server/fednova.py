

import numpy as np
import copy
import gc 
import torch

from collections import OrderedDict
from torch import nn
from dataset.partition import partition_data, get_dataloader
from models.model import get_model
from client.fednova import FedNovaClient
from utils.util import eval_test

class FedNovaServer(object):
    def __init__(self,
                 device: torch.device, 
                 model: str = 'resnet9',
                 dataset: str = 'vehicle10',
                 datadir: str = '../data',
                 num_users: int = 100,
                 partition: str = 'percentage20',
                 local_view: bool = True,
                 local_bs: int = 32,
                 local_ep: int = 10,
                 batch_size: int = 128,
                 lr: float = 0.01,
                 momentum: float = 0.5,
                 rounds: int = 100,
                 frac: float = 0.2,
                 print_freq: int = 10,
                 ) -> None:

        self.device = device
        self.model = model
        self.dataset = dataset
        self.datadir = datadir
        self.num_users = num_users
        self.partition = partition
        self.local_view = local_view
        self.local_bs = local_bs
        self.batch_size = batch_size
        self.local_ep = local_ep
        self.lr = lr
        self.momentum = momentum
        self.rounds = rounds
        self.frac = frac
        self.print_freq = print_freq

        self.net_glob: nn.Module = None
        self.w_glob: OrderedDict[str, torch.Tensor] = None

        self.__partition_data__()

        self.__build_model__()

        self.__init_clients__()

    def __partition_data__(self,):
        
        self.X_train, self.y_train, self.X_test, self.y_test, self.net_dataidx_map, self.net_dataidx_map_test, \
        self.traindata_cls_counts, self.testdata_cls_counts = partition_data(self.dataset, 
        self.datadir, self.partition, self.num_users, local_view=self.local_view)

        self.train_dl_global, self.test_dl_global, self.train_ds_global, self.test_ds_global = get_dataloader(self.dataset,
                                                                                        self.datadir,
                                                                                        self.batch_size,
                                                                                        32)

        print("len train_ds_global:", len(self.train_ds_global))
        print("len test_ds_global:", len(self.test_ds_global))

    def __build_model__(self,):
        print(f'MODEL: {self.model}, Dataset: {self.dataset}')

        self.users_model = []

        for net_i in range(-1, self.num_users):
           
            net = get_model(self.model)
            net.to(self.device)

            if net_i == -1: 
                self.net_glob = copy.deepcopy(net)
                self.initial_state_dict = copy.deepcopy(self.net_glob.state_dict())
                self.server_state_dict = copy.deepcopy(self.net_glob.state_dict())
            else:
                self.users_model.append(copy.deepcopy(net))
                self.users_model[net_i].load_state_dict(self.initial_state_dict)

        print(self.net_glob)

    def __init_clients__(self,):
        self.clients = []

        for idx in range(self.num_users):
            
            dataidxs = self.net_dataidx_map[idx]
            if self.net_dataidx_map_test is None:
                dataidx_test = None 
            else:
                dataidxs_test = self.net_dataidx_map_test[idx]

            #print(f'Initializing Client {idx}')

            train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(self.dataset, 
                                                                            self.datadir, self.local_bs, 32, 
                                                                            dataidxs, 
                                                                            dataidxs_test=dataidxs_test)
            
            self.clients.append(FedNovaClient(idx, copy.deepcopy(self.users_model[idx]), self.local_bs, self.local_ep, 
                    self.lr, self.momentum, self.device, train_dl_local, test_dl_local))


    def start(self,):

        loss_train = []
        global_acc = []

        clients_best_acc = [0 for _ in range(self.num_users)]
        w_locals, loss_locals = [], []

        ckp_avg_tacc = []
        ckp_avg_best_tacc = []

        users_best_acc = [0 for _ in range(self.num_users)]
        best_glob_acc = 0

        self.w_glob = copy.deepcopy(self.initial_state_dict)
        print_flag = False
        for iteration in range(self.rounds):
                
            m = max(int(self.frac * self.num_users), 1)
            idxs_users = np.random.choice(range(self.num_users), m, replace=False)
            
            print(f'###### ROUND {iteration+1} ######')
            print(f'Clients {idxs_users}')
                
            a_list = []
            d_list = []
            n_list = []
            for idx in idxs_users:
                
                self.clients[idx].set_state_dict(copy.deepcopy(self.w_glob)) 
                    
                loss, a_i, d_i = self.clients[idx].train(copy.deepcopy(self.w_glob), is_print=False)
                                
                a_list.append(a_i)
                d_list.append(d_i)
                n_i = len(self.net_dataidx_map[idx])
                n_list.append(n_i)
                
                loss_locals.append(copy.deepcopy(loss))        
            
            total_n = sum(n_list)
            
            d_total_round = copy.deepcopy(self.initial_state_dict)
            for key in d_total_round:
                d_total_round[key] = torch.zeros_like(self.initial_state_dict[key])
                
            for i in range(len(idxs_users)):
                d_para = d_list[i]
                for key in d_para:
                    d_total_round[key] = d_total_round[key].to(self.device) + d_para[key].to(self.device) * n_list[i] / total_n
            
            # update global model
            coeff = 0.0
            for i in range(len(idxs_users)):
                coeff = coeff + a_list[i] * n_list[i]/total_n

            updated_model = copy.deepcopy(self.w_glob)
            for key in updated_model:
                d_total_round[key].to(self.device)
                updated_model[key].to(self.device)
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    updated_model[key] = updated_model[key].to(self.device) - (coeff * d_total_round[key]).to(self.device)
            
            self.w_glob = copy.deepcopy(updated_model)
            self.net_glob.load_state_dict(copy.deepcopy(self.w_glob))
            _, acc = eval_test(self.net_glob, self.test_dl_global, self.device)
            if acc > best_glob_acc:
                best_glob_acc = acc 

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
                
            print('## END OF ROUND ##')
            template = 'Average Train loss {:.3f}'
            print(template.format(loss_avg))
            
            template = "Global Model Test Acc: {:.3f}, Global Model Best Test Acc: {:.3f}"
            print(template.format(acc, best_glob_acc))
            global_acc.append(acc)
            
            print_flag = False
            if iteration < 60:
                print_flag = True
            if iteration % self.print_freq == 0: 
                print_flag = True
                
            if print_flag:
                print('--- PRINTING ALL CLIENTS STATUS ---')
                current_acc = []
                for k in range(self.num_users):
                    loss, acc = self.clients[k].eval_test() 
                    current_acc.append(acc)
                    
                    if acc > clients_best_acc[k]:
                        clients_best_acc[k] = acc
                        
                    template = ("Client {:3d}, labels {}, count {}, best_acc {:3.3f}, current_acc {:3.3f} \n")
                    print(template.format(k, self.traindata_cls_counts[k], self.clients[k].get_count(),
                                        clients_best_acc[k], current_acc[-1]))
                    
                template = ("Round {:1d}, Avg current_acc {:3.3f}, Avg best_acc {:3.3f}")
                print(template.format(iteration+1, np.mean(current_acc), np.mean(clients_best_acc)))
                
                ckp_avg_tacc.append(np.mean(current_acc))
                ckp_avg_best_tacc.append(np.mean(clients_best_acc))
            
            print('----- Analysis End of Round -------')
            for idx in idxs_users:
                print(f'Client {idx}, Count: {self.clients[idx].get_count()}, Labels: {self.traindata_cls_counts[idx]}')
                
            loss_train.append(loss_avg)
            
            #break;
            ## clear the placeholders for the next round 
            loss_locals.clear()
            
            ## calling garbage collector 
            gc.collect()


    def eval(self,):
        test_loss = []
        test_acc = []
        train_loss = []
        train_acc = []

        for idx in range(self.num_users):        
            loss, acc = self.clients[idx].eval_test()
                
            test_loss.append(loss)
            test_acc.append(acc)
            
            loss, acc = self.clients[idx].eval_train()
            
            train_loss.append(loss)
            train_acc.append(acc)

        test_loss = sum(test_loss) / len(test_loss)
        test_acc = sum(test_acc) / len(test_acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_acc) / len(train_acc)

        print(f'Train Loss: {train_loss}, Test_loss: {test_loss}')
        print(f'Train Acc: {train_acc}, Test Acc: {test_acc}')

        self.net_glob.load_state_dict(copy.deepcopy(self.w_glob))
        _, acc = eval_test(self.net_glob, self.test_dl_global, self.device)

        template = "Global Model Test Acc: {:.3f}"
        print(template.format(acc))