import os
import json
from torch.utils.data import Dataset
from typing import Optional, Callable
from PIL import Image

class Vehicle10Dataset(Dataset): 
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