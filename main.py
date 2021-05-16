import argparse
from utils import process_data
from Net import *

if __name__ == "__main__":
    import warnings
    import time
    start = time.time()
    # ArgumentParser参数解析器，描述它做了什么
    parser = argparse.ArgumentParser(description='DSCNet')
    # add_argument函数来增加参数
    parser.add_argument('--db', default='coil20',
                        choices=['coil20', 'orl'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    # parse_args获取解析的参数
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x, y, num_sample, kmeansNum, channels, kernels, epochs, \
    weight_coef, weight_selfExp, alpha, dim_subspace, weight_cc, ro, SC_method = process_data(args)
    dscnet = SDSC(num_sample=num_sample, channels=channels, kernels=kernels, kmeansNum=kmeansNum)
    dscnet.to(device)
    ae_state_dict = torch.load('pretrained_weights_original/%s.pkl' % args.db)
    dscnet.ae.load_state_dict(ae_state_dict)
    print("Pretrained ae weights are loaded successfully.")
    train(dscnet, x, y, epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
          weight_cc=weight_cc, alpha=alpha, dim_subspace=dim_subspace,
          ro=ro, show=args.show_freq, SC_method=SC_method, device=device)
    torch.save(dscnet.state_dict(), args.save_dir + '/%s-model.ckp' % args.db)