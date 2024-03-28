
from .dataset import Vehicle10Dataset
from torchvision import transforms
from torch.utils import data

def load_data_vehicle10(root, batch_size, resize=(32, 32)):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    
    vehicle10_train = Vehicle10Dataset(
        root=root, split='train', transform=trans)

    vehicle10_test = Vehicle10Dataset(
        root=root, split='test', transform=trans) 
    
    return (
        data.DataLoader(vehicle10_train, batch_size, shuffle=True,
                       num_workers=4),
        data.DataLoader(vehicle10_test, batch_size, shuffle=False,
                       num_workers=4)
    )