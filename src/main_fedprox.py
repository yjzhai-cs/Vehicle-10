
import torch
from utils.options import args_parser
from server.fedprox import FedProxServer


if __name__ == '__main__':
    args = args_parser()
    print(str(args))

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 

    sever = FedProxServer(
        device = args.device,
        model = args.model,
        dataset = args.dataset,
        datadir = args.datadir,
        num_users = args.num_users,
        partition = args.partition,
        local_view = args.local_view,
        local_bs = args.local_bs,
        local_ep = args.local_ep,
        batch_size = args.batch_size,
        lr = args.lr,
        momentum = args.momentum,
        rounds = args.rounds,
        frac = args.frac,
        print_freq = args.print_freq,
        mu = args.mu,
    )

    sever.start()
    sever.eval()