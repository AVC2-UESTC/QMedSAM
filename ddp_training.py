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
from quantized_segment_anything import build_quant_sam, MedSAM
from monai.losses import DiceLoss
from functools import partial
from logging import Logger
from tqdm import tqdm

GPU_NUM = 1
GPU_RANK = [0]


def train(
        rank: int,
        world_size: int,
        master_port: int = 10309,
        train_ie: bool = True,
        train_pe: bool = False,
        train_md: bool = False,
        bits: int = 8,
        load_ckpt: str = None,
        save_ckpt: str = None,
        dataset: str = None,
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
    sam = build_quant_sam(bits, load_ckpt)
    medsam = MedSAM(sam, train_ie, train_pe, train_md).to(device).train()
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
                eta_min=lr/100,
                last_epoch=warmup_epochs,
            )
        ],
        milestones=[warmup_epochs],
    )
    num_epochs = warmup_epochs + cosine_epochs + 1

    # Build dataset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if GPU_NUM > 1 else None
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, drop_last=True)

    seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = BCEWithLogitsLoss(reduction='mean')
    best_loss = 1e5

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        if GPU_NUM > 1:
            sampler.set_epoch(epoch)
        epoch_loss = 0
        for image, gt, boxes in tqdm(loader) if rank == 0 else loader:
            optimizer.zero_grad()
            image, gt = image.to(device), gt.to(device)
            if GPU_NUM > 1:
                pd = ddp_model(image, boxes)
            else:
                pd = medsam(image, boxes)
            loss = seg_loss(pd, gt) + ce_loss(pd, gt.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        epoch_loss /= len(loader)
        logger.info(f'Epoch {epoch + 1:02}, rank {rank}, train loss = {epoch_loss:.3f}')
        if rank == 0 and best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save((ddp_model.module if GPU_NUM > 1 else medsam).state_dict(), save_ckpt)
        dataset.shuffle()

    # Clean up DDP
    if GPU_NUM > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    load_fp = r'./ckpt/QuantIE/8bits_v1.pth'
    save_path = './ckpt/QuantIE'
    save_name = f'8bits_v2'
    os.makedirs(save_path, exist_ok=True)
    log = sdk.create_logger(os.path.join(save_path, save_name + '.log'))
    # ds = sdk.GatherModalityDataset('/datadisk2/luhaisheng/dataset/MedSAMLaptop/train')
    ds = sdk.GatherModalityDataset('../dataset/MedSAMLaptop/train', logger=log)
    cudnn.benchmark = True
    train_fn = partial(
        train,
        world_size=GPU_NUM,
        save_ckpt=os.path.join(save_path, save_name + '.pth'),
        load_ckpt=load_fp,
        dataset=ds,
        logger=log,
    )  # spawn can only pass args, which is much more inconvenient than passing kwargs
    if GPU_NUM > 1:
        spawn(train_fn, nprocs=GPU_NUM, join=True)
    else:
        train_fn(0)
