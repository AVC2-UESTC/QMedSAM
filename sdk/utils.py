import logging
import pickle


def create_logger(log_fp, logger_name: str = 'inspector', ddp: bool = True) -> logging.Logger:
    """
    Create logger.
    :param log_fp: file pointer to logger
    :param logger_name: specified name for logger
    :param ddp: whether to use custom logger
    :return: customized logger
    """
    if ddp:
        logger = SimpleLogger(log_fp)
    else:
        logger = logging.getLogger(logger_name)
        sh = logging.StreamHandler()
        fh = logging.FileHandler(log_fp, mode='a')
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        logger.setLevel(logging.INFO)
        sh.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


class SimpleLogger:
    def __init__(self, log_fp: str):
        self.log_fp = log_fp

    def info(self, msg: str):
        import datetime
        msg = f'[{datetime.datetime.now()}] {msg}'
        print(msg)
        with open(self.log_fp, 'a') as log_file:
            log_file.write(msg + '\n')


class Averager:
    def __init__(self):
        self.cnt = 0
        self.tot = 0

    def avg(self):
        return self.cnt / self.tot if self.tot > 0 else 0

    def push(self, _x):
        self.cnt += _x
        self.tot += 1

    def reset(self):
        self.cnt = 0
        self.tot = 0

    def merge(self, _other):
        self.cnt += _other.cnt
        self.tot += _other.tot

    def save(self, fp):
        with open(fp, 'wb') as f:
            pickle.dump((self.cnt, self.tot), f)

    def load(self, fp):
        with open(fp, 'rb') as f:
            self.cnt, self.tot = pickle.load(f)


def apply_boxes(boxes, oh, ow, nh, nw):
    boxes = apply_coords(boxes.reshape(-1, 2, 2), oh, ow, nh, nw)
    return boxes.reshape(-1, 4)


def apply_coords(coords, oh, ow, nh, nw):
    from copy import deepcopy
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (nw / ow)
    coords[..., 1] = coords[..., 1] * (nh / oh)
    return coords


def cal_iou(result, reference):
    import torch
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    iou = intersection.float() / union.float()
    return iou.unsqueeze(1)


def load_yaml(yaml_fp: str):
    import yaml
    with open(yaml_fp, 'r') as f:
        return yaml.safe_load(f)
