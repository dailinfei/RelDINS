import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str, default='FB15K')    # 数据集
    arg.add_argument('-model', type=str, default='TransE')  # 选择模型
    arg.add_argument('-batch_size', type=int, default=1024)  # 批大小
    arg.add_argument('-margin', type=float, default=6.0)    # 正负样本的margin
    arg.add_argument('-dim', type=int, default=128)      # 嵌入维度
    arg.add_argument('-epoch', type=int, default=1000)    # 训练轮数
    arg.add_argument('-save', type=str)
    arg.add_argument('-img_dim', type=int, default=4096)
    arg.add_argument('-neg_num', type=int, default=1)
    arg.add_argument('-learning_rate', type=float, default=0.001)
    arg.add_argument('-lrg', type=float, default=0.001)
    arg.add_argument('-lrd', type=float, default=0.001)
    # arg.add_argument('-adv_temp', type=float, default=2.0)
    arg.add_argument('-adv_temp', type=float, default=2.5)
    arg.add_argument('-visual', type=str, default='random')
    arg.add_argument('-seed', type=int, default=3407)   # 42
    arg.add_argument('-missing_rate', type=float, default=0.8)
    arg.add_argument('-postfix', type=str, default='')
    arg.add_argument('-con_temp', type=float, default=0)
    arg.add_argument('-lamda', type=float, default=0)
    arg.add_argument('-mu', type=float, default=0)
    arg.add_argument('-adv_num', type=int, default=1)
    arg.add_argument('-disen_weight', type=float, default=0.01)
    arg.add_argument('-miss_type', type=str, default=None)
    arg.add_argument('-miss_prop', type=float, default=None)

    # 以下为扩散模型参数
    arg.add_argument('--use_diffusion', action='store_true', default=False)
    arg.add_argument('-T', type=int, default=10)
    arg.add_argument('-channel', type=int, default=128)
    arg.add_argument('-channel_mult', type=int, nargs='+', default=[1, 2, 3, 4])
    arg.add_argument('-attn', type=int, nargs='+', default=[2])
    arg.add_argument('-num_res_blocks', type=int, default=2)
    arg.add_argument('-dropout', type=float, default=0.1)

    arg.add_argument('-beta_1', type=float, default=1e-4)
    arg.add_argument('-beta_T', type=float, default=0.02)
    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
