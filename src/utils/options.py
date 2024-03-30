import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--trial', type=int, default=1, help="the trial number")
    parser.add_argument('--rounds', type=int, default=500, help="rounds of training")

    parser.add_argument('--local_bs', type=int, default=10, help="local training batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--size', type=int, default=32, help="the img size")
    parser.add_argument('--batch_size', type=int, default=64, help="test batch size")

    parser.add_argument('--model', type=str, default='lenet5', help='model name')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # federated arguments
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--nclass', type=int, default=2, help="classes or shards per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--mu', type=float, default=0.001, help="FedProx Regularizer")

    parser.add_argument('--dataset', type=str, default='vehicle10', 
                        help="name of dataset: vehicle10")
    parser.add_argument('--datadir', type=str, default='../data/', help='data directory')
    parser.add_argument('--partition', type=str, default='partition20', help='method of partitioning')
    parser.add_argument('--print_freq', type=int, default=10, help="printing frequency during training rounds")
    parser.add_argument('--local_view', action='store_true', help='whether from client perspective or not')



    args = parser.parse_args()
    return args