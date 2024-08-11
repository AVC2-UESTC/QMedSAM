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
from torchvision.transforms import Resize
from quantized_segment_anything import QuantLiteMedSAM, build_sam, MedSAM
from monai.losses import DiceLoss
from functools import partial
from logging import Logger
from tqdm import tqdm


# control panel
CONFIG_YAML = os.path.join('config', 'finaltune_quantize_iemd-2.yaml')
PORT = 10311
GPU_RANK = [2, 7, 8, 9]
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
        teacher_ckpt: str = None,
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
    litemedsam = QuantLiteMedSAM(
        ckpt=load_ckpt,
        quant_ie=quant_ie,
        quant_md=quant_md,
        bits=bits,
        quant_conv=quant_conv,
        mode='with_ie_output',
    ).to(device).train()
    litemedsam.modify_grad(train_ie, train_pe, train_md)
    if teacher_ckpt is not None:
        medsam = MedSAM(
            sam_model=build_sam(teacher_ckpt),
            grad_ie=False,
            grad_pe=False,
            grad_md=False,
            upscale_mask=False,
            mode='with_ie_output',
        ).to(device).eval()
    else:
        medsam = None
    params = []
    if GPU_NUM > 1:
        ddp_litemedsam = DistributedDataParallel(
            module=litemedsam,
            device_ids=[GPU_RANK[rank]],
            output_device=device,
            find_unused_parameters=True,
        )
        if train_ie:
            params += list(ddp_litemedsam.module.image_encoder.parameters())
        if train_pe:
            params += list(ddp_litemedsam.module.prompt_encoder.parameters())
        if train_md:
            params += list(ddp_litemedsam.module.mask_decoder.parameters())
    else:
        ddp_litemedsam = None
        if train_ie:
            params += list(litemedsam.image_encoder.parameters())
        if train_pe:
            params += list(litemedsam.prompt_encoder.parameters())
        if train_md:
            params += list(litemedsam.mask_decoder.parameters())

    # Init SGD & WarmUpCosineAnnealingLR
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
        sampler = None
        loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, drop_last=True)

    seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = BCEWithLogitsLoss(reduction='mean')
    mse_loss = torch.nn.MSELoss(reduction='mean')

    if rank == 0:
        logger.info('Start training')
    for epoch in range(num_epochs):
        if GPU_NUM > 1:
            sampler.set_epoch(epoch)
        train_loss = 0
        for image, gt, boxes in tqdm(loader) if rank == 0 else loader:
            optimizer.zero_grad()
            image, gt = image.to(device), gt.to(device)
            boxes = boxes[:, None, :].to(device)
            if GPU_NUM > 1:
                pd_student, student_embed = ddp_litemedsam(image, boxes)
            else:
                pd_student, student_embed = litemedsam(image, boxes)
            loss = seg_loss(pd_student, gt) + ce_loss(pd_student, gt.float())
            if medsam is not None:
                image_resizer = Resize([1024] * 2)
                pd_teacher, teacher_embed = medsam(image_resizer(image), boxes * 4)
                dist_loss = sdk.cal_iou(pd_teacher > 0, gt.bool()).mean() * mse_loss(student_embed, teacher_embed)
                loss = loss + dist_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        train_loss /= len(loader)
        if rank == 0:
            logger.info(f'Epoch {epoch+1:02}, rank {rank}, train loss = {train_loss:.3f}')
        state_dict = (ddp_litemedsam.module if GPU_NUM > 1 else litemedsam).state_dict()
        torch.save(state_dict, save_ckpt + f'.e{epoch+1:02}.pth')
        dataset_train.shuffle()

    # Clean up DDP
    if GPU_NUM > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    config = sdk.load_yaml(CONFIG_YAML)
    load_model_fp = config['load_model_fp']
    teacher_model_fp = config['teacher_model_fp']  # specify None to disable distillation
    save_model_dir = config['save_model_dir']
    save_model_name = config['save_model_name']
    train_dataset_dir = config['train_dataset_dir']

    os.makedirs(save_model_dir, exist_ok=True)
    train_logger = sdk.create_logger(os.path.join(save_model_dir, save_model_name + '.log'))
    train_ds = sdk.GatherModalityDataset(
        train_dataset_dir,
        image_size=config['image_size'],
        bbox_shift=config['bbox_shift'],
        mod_sample=None,  # fetch all samples
        augment=config['augment'],
        logger=train_logger,
    )
    sampler_strategy = config['sampler_strategy']
    if sampler_strategy == 'Min':
        target_samples = min(train_ds.mod_samples())
        train_logger.info(f'Adjust num_samples of all the modalities to {target_samples}')
        train_ds.modify_mod_sample(target_samples)
    elif sampler_strategy == 'OneTenth':
        all_samples = train_ds.mod_samples()
        min_samples = min(all_samples)
        target_samples = [max(int(sp / 10), min_samples) for sp in all_samples]
        train_logger.info(f'Adjust num_samples of all the modalities to {target_samples}')
        train_ds.modify_mod_sample(target_samples)
    elif sampler_strategy == 'All':
        pass
    else:
        raise NotImplementedError(f'Sampler strategy {sampler_strategy} not implemented.')

    cudnn.benchmark = True
    train_fn = partial(
        train,
        master_port=PORT,
        world_size=GPU_NUM,
        quant_ie=config['quant_ie'],
        quant_md=config['quant_md'],
        train_ie=config['train_ie'],
        train_pe=config['train_pe'],
        train_md=config['train_md'],
        save_ckpt=os.path.join(save_model_dir, save_model_name),
        load_ckpt=load_model_fp,
        teacher_ckpt=teacher_model_fp,
        dataset_train=train_ds,
        warmup_epochs=config['warmup_epochs'],
        cosine_epochs=config['cosine_epochs'],
        logger=train_logger,
        batch_size=config['batch_size'],
    )  # spawn can only pass args, which is much more inconvenient than passing kwargs
    if GPU_NUM > 1:
        spawn(train_fn, nprocs=GPU_NUM, join=True)
    else:
        train_fn(0)