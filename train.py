import os
import argparse
import random
import sys

from torchpack import distributed as dist
import gc
import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import InferenceRunner, MaxSaver, Saver
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torchpack.environ import auto_set_run_dir, set_run_dir
from core.builder.trainBuilder import SensatUrbanTrainer
from core.builder.builder import make_dataset, make_model, make_criterion, make_optimizer, make_scheduler
from core.callbacks import MeanIoU
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='gpu nums')
    parser.add_argument('--run-dir', required=False, metavar='DIR', help='run directory')
    args, opts = parser.parse_known_args()

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    configs.load("./configs/default.yaml", recursive=True)
    configs.update(opts)

    os.environ['MASTER_HOST'] = 'localhost:8888'
    # os.environ["RANK"] = "0"
    os.environ['WORLD_SIZE'] = '1'

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dist.init()
    local_rank = dist.local_rank()
    # print(local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", 0)

    torch.backends.cudnn.benchmark = True

    seed = configs.train.seed + dist.rank() * configs.workers_per_gpu * configs.train.num_epochs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    datasets = make_dataset()
    # datasets['train'].__getitem__(0)['lidar']
    # exit()
    dataflow = {}

    for split in datasets:
        sampler = torch.utils.data.distributed.DistributedSampler(
            datasets[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            datasets[split],
            batch_size=configs.train.batch_size,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=False,
            collate_fn=datasets[split].collate_fn)

    model = make_model().to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # torch.distributed.init_process_group(backend="nccl")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank)

    criterion = make_criterion()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)

    trainer = SensatUrbanTrainer(model=model,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 scheduler=scheduler,
                                 num_workers=configs.workers_per_gpu,
                                 seed=seed,
                                 amp_enabled=configs.amp_enabled)
    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.train.num_epochs,
        callbacks=[
                      InferenceRunner(
                          dataflow[split],
                          callbacks=[
                              MeanIoU(name=f'iou/{split}',
                                      num_classes=configs.data.num_classes)
                          ],
                      ) for split in ['test']
                  ] + [
                      MaxSaver('iou/test'),
                      Saver(),
                  ])
