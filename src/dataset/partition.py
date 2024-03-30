import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from .dataset import Vehicle10_truncated


def load_vehicle10_data(datadir, resize=(32, 32)):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    vehicle10_train_ds = Vehicle10_truncated(root=datadir, train=True, transform=trans, download=False)
    vehicle10_test_ds = Vehicle10_truncated(root=datadir, train=False, transform=trans, download=False)

    X_train, y_train = vehicle10_train_ds.data, vehicle10_train_ds.target
    X_test, y_test = vehicle10_test_ds.data, vehicle10_test_ds.target

    X_train = np.array(X_train, dtype=object)
    y_train = np.array(y_train, dtype=object)
    X_test = np.array(X_test, dtype=object)
    y_test = np.array(y_test, dtype=object)

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts

def partition_data(dataset, datadir, partition, n_parties, local_view=False):
    if dataset == 'vehicle10':
        X_train, y_train, X_test, y_test = load_vehicle10_data(datadir, resize=(32, 32))
        n_classes = 10
    else:
        raise RuntimeError(f'{dataset} dataset is undefined')
    
    if partition[:10] == 'percentage':
        percentage = eval(partition[10:]) / 100
        num =  max(round(n_classes * percentage), 1)
        K = n_classes

        print(f'K: {K}, num: {num}')
        
        if num == K:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(K)]
            contain=[]
            for i in range(n_parties):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1        
    else:
        raise RuntimeError(f'{partition} Non-IID setting is undefined')

    print(f'partition: {partition}')
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    print('Data statistics Train:\n %s \n' % str(traindata_cls_counts))
    
    if local_view:
        net_dataidx_map_test = {i: [] for i in range(n_parties)}
        for k_id, stat in traindata_cls_counts.items():
            labels = list(stat.keys())
            for l in labels:
                idx_k = np.where(y_test==l)[0]
                net_dataidx_map_test[k_id].extend(idx_k.tolist())

        testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test)
        print('Data statistics Test:\n %s \n' % str(testdata_cls_counts))
    else: 
        net_dataidx_map_test = None 
        testdata_cls_counts = None 

    return (X_train, y_train, X_test, y_test, net_dataidx_map, net_dataidx_map_test, traindata_cls_counts, testdata_cls_counts)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, net_id=None, dataidxs_test=None,
                  same_size=False, target_transform=None):
    if dataset == 'vehicle10':
        dl_obj = Vehicle10_truncated
        transform_train = transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, 
                                  target_transform=target_transform, download=True)
        test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, 
                                 target_transform=target_transform, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)
    else:
        raise RuntimeError(f'{dataset} dataset is undefined')
    
    return train_dl, test_dl, train_ds, test_ds