import torch
from mmkgc.config import Tester
from mmkgc.config.RelDINSTrainerGP import RelDINSTrainerGP
from mmkgc.module.loss import SigmoidLoss

from mmkgc.module.model.RelDINSRotatE import RelDINSRotatE

from mmkgc.module.ns.DiffusionModel import UNet
from mmkgc.module.ns.interpolation_sample import InterpolationSampler

from mmkgc.module.strategy import NegativeSamplingGP
from mmkgc.data import TrainDataLoader, TestDataLoader


from args import get_args

if __name__ == "__main__":
    args = get_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_dataloader = TrainDataLoader(
        in_path="./benchmarks/" + args.dataset + '/',
        batch_size=args.batch_size,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=args.neg_num,
        neg_rel=0
    )
    # dataloader for test
    test_dataloader = TestDataLoader(
        "./benchmarks/" + args.dataset + '/', "link")
    img_emb = torch.load('./embeddings/' + args.dataset + '-visual.pth')
    text_emb = torch.load('./embeddings/' + args.dataset + '-textual.pth')


    # define the model
    kge_score = RelDINSRotatE(
        ent_tot=train_dataloader.get_ent_tot(),     # 实体总数
        rel_tot=train_dataloader.get_rel_tot(),     # 关系总数
        dim=args.dim,                             # 嵌入维度
        margin=args.margin,                         # 边界
        epsilon=2.0,
        img_emb=img_emb,
        text_emb=text_emb,
        train2id_path='./benchmarks/' + args.dataset + '/train2id.txt'
    )
    print(kge_score)
    # define the loss function
    model = NegativeSamplingGP(
        model=kge_score,
        loss=SigmoidLoss(adv_temperature=args.adv_temp),
        batch_size=train_dataloader.get_batch_size(),
        regul_rate=0.00001
    )


    diffusion_model = UNet(
        T=args.T,
        ch=args.channel,
        ch_mult=args.channel_mult,
        attn=args.attn,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout
    )
    sampler = InterpolationSampler(
        ent_tot=train_dataloader.get_ent_tot(),
        in_path="./benchmarks/" + args.dataset + '/',
        diffusion_model=diffusion_model,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        T=args.T
    )


    # train the model
    trainer = RelDINSTrainerGP(
        model=model,
        data_loader=train_dataloader,
        train_times=args.epoch,
        alpha=args.learning_rate,
        use_gpu=True,
        opt_method='Adam',
        lrg=args.lrg,
        mu=args.mu,
        g_epoch=100,
        sampler=sampler,
        use_diffusion=args.use_diffusion
    )

    trainer.run()
    kge_score.save_checkpoint(args.save)

    # test the model
    kge_score.load_checkpoint(args.save)
    tester = Tester(model=kge_score, data_loader=test_dataloader, use_gpu=True)
    tester.run_link_prediction(type_constrain=False)
