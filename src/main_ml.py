
import torch
import torch.nn.functional as F
from torch import nn
from utils.options import args_parser
from dataset.dataloader import load_data_vehicle10
from models.model import get_model

def test(net, ldr_test, device):
    net.to(device)
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in ldr_test:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
    test_loss /= len(ldr_test.dataset)
    accuracy = 100. * correct / len(ldr_test.dataset)
    return test_loss, accuracy

def train(args):
    print('load data ...')
    ldr_train, ldr_test = load_data_vehicle10(root='../data', batch_size=args.local_bs, resize=(args.size, args.size))

    print('training on', args.device)
    net = get_model(args.model)
    net.to(args.device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    loss_func = nn.CrossEntropyLoss()
    net.train()
    
    epoch_loss = []
    for epoch in range(args.rounds):
        print(f'training for {epoch + 1} itertions ...')

        batch_loss = []
        for batch_idx, (images, labels) in enumerate(ldr_train):
            images, labels = images.to(args.device), labels.to(args.device)
            net.zero_grad()

            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward() 
                        
            optimizer.step()
            batch_loss.append(loss.item())
                
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    train_loss = sum(epoch_loss) / len(epoch_loss)
    
    print('test ..')
    test_loss, test_acc = test(net, ldr_test, args.device)

    print(f'train loss {train_loss:.3f}, test loss {test_loss:.3f}, test_acc {test_acc:.3f}')


if __name__ == "__main__":
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu)

    print(str(args))

    train(args)