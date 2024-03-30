import os
import json
from torch.utils.data import Dataset
from typing import Optional, Callable
from PIL import Image

class Vehicle10Dataset(Dataset): 
    """Vehicle10Dataset for Machine Learning."""
    def __init__(self, 
                root: str,
                split: str = 'train',
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                ) -> None:
        
        self.root = root
        
        if not os.path.exists(os.path.join(root, 'vehicle-10')):
            data_dir = os.path.join(root, 'vehicle-10')
            raise RuntimeError(f'{data_dir} is not a directory')
    
        self.meta_info = None
        if split == 'train':
            with open(os.path.join(root, 'vehicle-10', 'train_meta.json'), 'r') as file:
                self.meta_info = json.load(file)
        elif split == 'test':
            with open(os.path.join(root, 'vehicle-10', 'valid_meta.json'), 'r') as file:
                self.meta_info = json.load(file)
        else:
            raise RuntimeError(f'{split} is undefined')
        
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = os.path.join(self.root, 'vehicle-10', self.meta_info['path'][index])
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.meta_info['label'][index]
        # torch.tensor(label)
        
        return img, label

    def __len__(self):
        return len(self.meta_info['path'])
    

class Vehicle10_truncated(Dataset):
    """Vehicle10_truncated for Federated Learning."""
    def __init__(self, 
                 root, 
                 dataidxs=None, 
                 train=True, 
                 transform=None, 
                 target_transform=None, 
                 download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self,):
        split = 'train' if self.train else 'test'

        vehicle10_dataobj = Vehicle10Dataset(root=self.root, split=split, transform=self.transform, target_transform=self.target_transform)
        data, target = [], []
        if self.dataidxs is None:
            self.dataidxs = [i for i in range(len(vehicle10_dataobj))]
        
        for idx in self.dataidxs:
            data.append(vehicle10_dataobj[idx][0])
            target.append(vehicle10_dataobj[idx][1])

        return data, target

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self,):
        return len(self.data)