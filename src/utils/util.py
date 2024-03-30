import torch
import torch.nn.functional as F

def eval_test(net, ldr_test, device): 
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