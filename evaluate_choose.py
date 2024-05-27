import multiprocessing as mp
import os
import sdk
import queue
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.nn.parallel import DistributedDataParallel
from torch.multiprocessing import spawn
from monai.losses import DiceLoss
from quantized_segment_anything import QuantLiteMedSAM
from functools import partial


class SplitDataset(Dataset):
    def __init__(self, parts: int, rank: int, source_dataset: Dataset):
        from copy import deepcopy
        block = len(source_dataset) // parts
        assert rank < parts
        self.start = rank * block
        self.end = min(len(source_dataset), (rank + 1) * block - 1)
        self.source_dataset = deepcopy(source_dataset)

    def __len__(self):
        return self.end - self.start + 1

    def __getitem__(self, item):
        item = item + self.start
        return self.source_dataset[item]


@torch.no_grad()
def evaluate(model, dataset, device, batch_size, q):
    res = sdk.Averager()
    device = torch.device(f'cuda:{device}')
    loader = DataLoader(dataset, batch_size, shuffle=False)
    model = model.eval().to(device)
    seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = BCEWithLogitsLoss(reduction='mean')
    for image, gt, boxes in tqdm(loader):
        image, gt = image.to(device), gt.to(device)
        boxes = boxes[:, None, :].to(device)
        pd = model(image, boxes)
        loss = seg_loss(pd, gt) + ce_loss(pd, gt.float())
        res.push(loss.item())
    q.put(res)


def ddp_evaluate(rank, world_size, master_port, batch_size, checkpoint, datasets, devices, model_kwargs):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    backend = dist.Backend.GLOO if os.name.startswith('nt') else dist.Backend.NCCL
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    device = torch.device(f'cuda:{devices[rank]}')
    model = QuantLiteMedSAM(checkpoint, **model_kwargs).eval().to(device)
    seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    ce_loss = BCEWithLogitsLoss(reduction='mean')
    loader = DataLoader(datasets[rank], batch_size, shuffle=False)
    res = sdk.Averager()
    with torch.no_grad():
        for image, gt, boxes in tqdm(loader) if rank == 0 else loader:
            image, gt = image.to(device), gt.to(device)
            boxes = boxes[:, None, :].to(device)
            pd = model(image, boxes)
            loss = seg_loss(pd, gt) + ce_loss(pd, gt.float())
            res.push(loss.item())
    # q.put(res)
    res.save(checkpoint + f'.tmp{rank}.pkl')
    dist.destroy_process_group()


if __name__ == '__main__':
    bs = 64
    ckpt_dir = './ckpt/'
    logger_fp = os.path.join(ckpt_dir, 'evaluate.log')
    logger = sdk.create_logger(logger_fp)
    eval_ds = sdk.GatherModalityDataset(
        dataset_fp='../dataset/MedSAMLaptop/split_set/eval',
        image_size=256,
        bbox_shift=10,
        mod_sample=None,
        shuffle=False,
        buffer_len=bs,
        logger=logger,
        augment=False,
    )
    model_param = {
        'quant_ie': True,
        'quant_md': True,
        'quant_conv': True,
        'bits': 8,
    }
    mods = eval_ds.modalities
    # mods = mods[2: 3]
    ckpt_list = [os.path.join(ckpt_dir, fn) for fn in os.listdir(ckpt_dir) if fn.endswith('.pth')]
    devs = [0]
    csv_save_fp = os.path.join(ckpt_dir, 'evaluate.csv')
    csv_str = 'CheckPoint,'
    for mod in mods:
        csv_str += f'{mod.name},'
    csv_str += '\n'
    for ckpt in ckpt_list:
        avg_mod_loss = sdk.Averager()
        msg = f'ckpt {os.path.basename(ckpt)} loss, '
        csv_str += os.path.basename(ckpt) + ','
        for mod in mods:
            cache_fp = ckpt + f'.eval.{mod.name}.pkl'
            mod_loss = sdk.Averager()
            if not os.path.exists(cache_fp):
                mod_loss_queue = queue.Queue()
                if len(devs) > 1:
                    # mod_loss_queue = mp.Queue()
                    split_datasets = [SplitDataset(len(devs), i, mod) for i in range(len(devs))]
                    eval_fn = partial(
                        ddp_evaluate,
                        world_size=len(devs),
                        master_port=10401,
                        batch_size=bs,
                        # model=medsam,
                        checkpoint=ckpt,
                        datasets=split_datasets,
                        devices=devs,
                        model_kwargs=model_param,
                        # q=mod_loss_queue,
                    )
                    spawn(eval_fn, nprocs=len(devs), join=True)
                    for i in range(len(devs)):
                        fp = ckpt + f'.tmp{i}.pkl'
                        ranki_res = sdk.Averager()
                        ranki_res.load(fp)
                        mod_loss_queue.put(ranki_res)
                        os.remove(fp)
                else:
                    medsam = QuantLiteMedSAM(ckpt, **model_param)
                    evaluate(medsam, mod, devs[0], bs, mod_loss_queue)
                while not mod_loss_queue.empty():
                    mod_loss.merge(mod_loss_queue.get())
                mod_loss.save(cache_fp)
            else:
                mod_loss.load(cache_fp)
            avg_mod_loss.push(mod_loss.avg())
            csv_str += f'{mod_loss.avg()},'
            msg += f'{mod.name}: {mod_loss.avg():.3f}, '
        csv_str += f'{avg_mod_loss.avg()}\n'
        msg += f'Average loss {avg_mod_loss.avg():.3f}'
        logger.info(msg)
    with open(csv_save_fp, 'w') as csv_file:
        csv_file.write(csv_str)
