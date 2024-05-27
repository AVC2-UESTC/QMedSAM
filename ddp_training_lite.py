import sdk
import os
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import spawn
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from quantized_segment_anything import QuantLiteMedSAM
from monai.losses import DiceLoss
from functools import partial
from logging import Logger
from tqdm import tqdm


GPU_RANK = [0]
GPU_NUM = len(GPU_RANK)


def train(
        rank: int,
        world_size: int,
        master_port: int = 10309,
        quant_ie: bool = True,
        quant_md: bool = False,
        train_ie: bool = True,
        train_pe: bool = False,
        train_md: bool = False,
        bits: int = 8,
        quant_conv: bool = True,
        load_ckpt: str = None,
        save_ckpt: str = None,
        dataset_train: sdk.GatherModalityDataset = None,
        batch_size: int = 1,
        lr: float = 1e-2,
        momentum: float = 9e-1,
        warmup_epochs: int = 10,
        cosine_epochs: int = 20,
        logger: Logger = None,
):
    # Setup DDP
    torch.manual_seed(19260817 + rank)
    if GPU_NUM > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(master_port)
        backend = dist.Backend.GLOO if os.name.startswith('nt') else dist.Backend.NCCL
        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )
    device = torch.device(f'cuda:{GPU_RANK[rank]}')

    # Init MedSAM
    medsam = QuantLiteMedSAM(load_ckpt, quant_ie, quant_md, bits, quant_conv).to(device).train()
    medsam.modify_grad(train_ie, train_pe, train_md)
    if GPU_NUM > 1:
        ddp_model = DistributedDataParallel(
            module=medsam,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=True,
        )
    else:
        ddp_model = None

    # Init SGD & WarmUpCosineAnnealingLR
    params = []
    model = ddp_model.module if GPU_NUM > 1 else medsam
    if train_ie:
        params += list(model.image_encoder.parameters())
    if train_pe:
        params += list(model.prompt_encoder.parameters())
    if train_md:
        params += list(model.mask_decoder.parameters())
    optimizer = SGD(params, lr=lr, momentum=momentum)
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[
            LinearLR(
                optimizer=optimizer,
                start_factor=1e-2,
                total_iters=warmup_epochs,
            ),
            CosineAnnealingLR(
                optimizer=optimizer,
                T_max=cosine_epochs,
                eta_min=lr/1000,
                last_epoch=warmup_epochs,
            )
        ],
        milestones=[warmup_epochs],
    )
    num_epochs = warmup_epochs + cosine_epochs + 1

    # Build dataset
    if GPU_NUM > 1:
        sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, drop_last=True)
        loader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=True)

    seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = BCEWithLogitsLoss(reduction='mean')

    if rank == 0:
        logger.info('Start training')
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        if GPU_NUM > 1:
            sampler.set_epoch(epoch)
        train_loss = 0
        for image, gt, boxes in tqdm(loader) if rank == 0 else loader:
            optimizer.zero_grad()
            image, gt = image.to(device), gt.to(device)
            boxes = boxes[:, None, :].to(device)
            if GPU_NUM > 1:
                pd = ddp_model(image, boxes)
            else:
                pd = medsam(image, boxes)
            loss = seg_loss(pd, gt) + ce_loss(pd, gt.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(loader)
        if rank == 0:
            logger.info(f'Epoch {epoch + 1:02}, rank {rank}, train loss = {train_loss:.3f}')
        torch.save((ddp_model.module if GPU_NUM > 1 else medsam).state_dict(), save_ckpt + f'.e{epoch}.pth')
        dataset_train.shuffle()

    # Clean up DDP
    if GPU_NUM > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # load_fp = './ckpt/8bits.e11.pth'
    load_fp = './ckpt/LiteMedSAM/lite_medsam.pth'
    save_path = './ckpt/LiteQIE900v2'
    save_name = f'8bits'
    train_fp = '../dataset/MedSAMLaptop/split_set/train'
    eval_fp = '../dataset/MedSAMLaptop/split_set/eval'
    os.makedirs(save_path, exist_ok=True)
    log = sdk.create_logger(os.path.join(save_path, save_name + '.log'))
    train_ds = sdk.GatherModalityDataset(
        train_fp,
        image_size=256,
        bbox_shift=10,
        mod_sample=900,
        augment=True,
        logger=log
    )
    # eval_ds = sdk.GatherModalityDataset(eval_fp, 256, 10, 90, log)
    # mod_samples = train_ds.mod_samples()
    # tar_size = 900
    # for i, ms in enumerate(mod_samples):
    #     if ms > 10000:
    #         mod_samples[i] = ms // 10
    # print(mod_samples)
    # train_ds.modify_mod_sample(mod_samples)
    # mod_samples = eval_ds.mod_samples()
    # for i, ms in enumerate(mod_samples):
    #     if ms > 100:
    #         mod_samples[i] = 100
    # eval_ds.modify_mod_sample(mod_samples)
    # print(f'train dataset has {len(train_ds)} 2d clips')
    # print(train_ds.mod_samples())
    # print(eval_ds.mod_samples())
    cudnn.benchmark = True
    train_fn = partial(
        train,
        master_port=10309,
        world_size=GPU_NUM,
        quant_ie=True,
        quant_md=False,
        train_ie=True,
        train_pe=False,
        train_md=False,
        bits=8,
        quant_conv=True,
        save_ckpt=os.path.join(save_path, save_name),
        load_ckpt=load_fp,
        dataset_train=train_ds,
        warmup_epochs=3,
        cosine_epochs=10,
        logger=log,
        batch_size=4,
    )  # spawn can only pass args, which is much more inconvenient than passing kwargs
    if GPU_NUM > 1:
        spawn(train_fn, nprocs=GPU_NUM, join=True)
    else:
        train_fn(0)
