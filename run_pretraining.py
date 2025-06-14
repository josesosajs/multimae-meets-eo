import os
import sys
import yaml
import time
import json
import torch
import utils
import argparse
import datetime
import warnings
import numpy as np
import torch.backends.cudnn as cudnn

from pathlib import Path
from multimae import multimae
from domain_config import CONFIG
from utils.optim_factory import create_optimizer
from engine_pretraining import train_one_epoch, get_model
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils.datasets import build_multimae_custom_pretraining_dataset
from utils.task_balancing import NoWeightingStrategy, UncertaintyWeightingStrategy

DOMAIN_CONF = CONFIG

def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MultiMAE meets EO pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size per GPU (default: %(default)s)')
    parser.add_argument('--epochs', default=1600, type=int, help='Number of epochs (default: %(default)s)')
    parser.add_argument('--save_ckpt_freq', default=20, type=int, help='Checkpoint saving frequency in epochs (default: %(default)s)')

    # Task parameters
    parser.add_argument('--in_domains', default='rgb-depth-semseg-ired-sired-ebands', type=str,
                        help='Input domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--out_domains', default='rgb-depth-semseg-ired-sired-ebands', type=str,
                        help='Output domain names, separated by hyphen (default: %(default)s)')
    parser.add_argument('--standardize_depth', action='store_true')
    parser.add_argument('--no_standardize_depth', action='store_false', dest='standardize_depth')
    parser.set_defaults(standardize_depth=False)
    parser.add_argument('--extra_norm_pix_loss', action='store_true')
    parser.add_argument('--no_extra_norm_pix_loss', action='store_false', dest='extra_norm_pix_loss')
    parser.set_defaults(extra_norm_pix_loss=True)

    # Model parameters
    parser.add_argument('--model', default='pretrain_multimae_base', type=str, metavar='MODEL',
                        help='Name of model to train (default: %(default)s)')
    parser.add_argument('--num_encoded_tokens', default=98, type=int,
                        help='Number of tokens to randomly choose for encoder (default: %(default)s)')
    parser.add_argument('--num_global_tokens', default=1, type=int,
                        help='Number of global tokens to add to encoder (default: %(default)s)')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Base patch size for image-like modalities (default: %(default)s)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Images input size for backbone (default: %(default)s)')
    parser.add_argument('--alphas', type=float, default=1.0, 
                        help='Dirichlet alphas concentration parameter (default: %(default)s)')
    parser.add_argument('--sample_tasks_uniformly', default=False, action='store_true',
                        help='Set to True/False to enable/disable uniform sampling over tasks to sample masks for.')

    parser.add_argument('--decoder_use_task_queries', default=True, action='store_true',
                        help='Set to True/False to enable/disable adding of task-specific tokens to decoder query tokens')
    parser.add_argument('--decoder_use_xattn', default=True, action='store_true',
                        help='Set to True/False to enable/disable decoder cross attention.')
    parser.add_argument('--decoder_dim', default=256, type=int,
                        help='Token dimension inside the decoder layers (default: %(default)s)')
    parser.add_argument('--decoder_depth', default=2, type=int,
                        help='Number of self-attention layers after the initial cross attention (default: %(default)s)')
    parser.add_argument('--decoder_num_heads', default=8, type=int,
                        help='Number of attention heads in decoder (default: %(default)s)')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: %(default)s)')

    parser.add_argument('--loss_on_unmasked', default=False, action='store_true',
                        help='Set to True/False to enable/disable computing the loss on non-masked tokens')
    parser.add_argument('--no_loss_on_unmasked', action='store_false', dest='loss_on_unmasked')
    parser.set_defaults(loss_on_unmasked=False)


    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: %(default)s)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer epsilon (default: %(default)s)')
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+', metavar='BETA', help='Optimizer betas (default: %(default)s)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='CLIPNORM', help='Clip gradient norm (default: %(default)s)')
    parser.add_argument('--skip_grad', type=float, default=None, metavar='SKIPNORM', help='Skip update if gradient norm larger than threshold (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay (default: %(default)s)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the weight decay. We use a cosine schedule for WD.""")
    parser.add_argument('--decoder_decay', type=float, default=None, help='decoder weight decay')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR', help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: %(default)s)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='Warmup learning rate (default: %(default)s)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='Lower lr bound for cyclic schedulers that hit 0 (default: %(default)s)')
    parser.add_argument('--task_balancer', type=str, default='none', help='Task balancing scheme. One out of [uncertainty, none] (default: %(default)s)')
    parser.add_argument('--balancer_lr_scale', type=float, default=1.0, help='Task loss balancer LR scale (if used) (default: %(default)s)')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--fp32_output_adapters', type=str, default='', help='Tasks output adapters to compute in fp32 mode, separated by hyphen.')

    # Augmentation parameters
    parser.add_argument('--hflip', type=float, default=0.5, help='Probability of horizontal flip (default: %(default)s)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic', help='Training interpolation (random, bilinear, bicubic) (default: %(default)s)')

    # Dataset parameters
    parser.add_argument('--data_path', default='data/data_1M_v001.h5', type=str, help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--classes_def', default='data/biome_labels.json', type=str)
    parser.add_argument('--splits_path', default='data/data_1M_v001_splits.json', type=str)

    # Misc.
    parser.add_argument('--output_dir', default='', help='Path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='Device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int, help='Random seed ')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=True)

    # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true', help='Log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default=None, type=str, help='Project name on wandb')
    parser.add_argument('--wandb_entity', default=None, type=str, help='User or team name on wandb')
    parser.add_argument('--wandb_run_name', default=None, type=str, help='Run name on wandb')
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

    # Distributed training parameters
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual, defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    return args


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if not args.show_user_warnings: warnings.filterwarnings("ignore", category=UserWarning)

    args.in_domains = args.in_domains.split('-')
    args.out_domains = args.out_domains.split('-')
    args.all_domains = list(set(args.in_domains) | set(args.out_domains))

    #Get model
    model = get_model(args, DOMAIN_CONF)

    if args.task_balancer == 'uncertainty': loss_balancer = UncertaintyWeightingStrategy(tasks=args.out_domains)
    else: loss_balancer = NoWeightingStrategy()

    tasks_loss_fn = {
        domain: DOMAIN_CONF[domain]['loss'](patch_size=args.patch_size, stride=DOMAIN_CONF[domain]['stride_level'])
        for domain in args.out_domains
    }

    # Add normalized pixel loss if specified
    if args.extra_norm_pix_loss:
        tasks_loss_fn['norm_rgb'] = DOMAIN_CONF['rgb']['loss'](patch_size=args.patch_size,
                                    stride=DOMAIN_CONF['rgb']['stride_level'],
                                    norm_pix=True)
    # Get dataset
    dataset_train = build_multimae_custom_pretraining_dataset(args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(dataset_train, 
            num_replicas=num_tasks, rank=sampler_rank, shuffle=True, drop_last=True)
        
        print("Sampler_train = %s" % str(sampler_train))
    
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_wandb: log_writer = utils.WandbLogger(args)
    else: log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True)

    model.to(device)

    loss_balancer.to(device)
    model_without_ddp = model
    loss_balancer_without_ddp = loss_balancer
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model = %s" % str(model_without_ddp))
    print(f"Number of params: {n_parameters / 1e6} M")

    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.blr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.distributed and args.task_balancer != 'none':
        loss_balancer = torch.nn.parallel.DistributedDataParallel(loss_balancer, device_ids=[args.gpu])
        loss_balancer_without_ddp = loss_balancer.module

    optimizer = create_optimizer(
        args, {'model': model_without_ddp, 'balancer': loss_balancer_without_ddp})
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    
    if args.weight_decay_end is None: args.weight_decay_end = args.weight_decay
    
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler)

    print(args)
    print(f"Start training for {args.epochs} epochs")
    
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed: data_loader_train.sampler.set_epoch(epoch)
        
        if log_writer is not None: log_writer.set_step(epoch * num_training_steps_per_epoch)
        
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            tasks_loss_fn=tasks_loss_fn,
            loss_balancer=loss_balancer,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_encoded_tokens=args.num_encoded_tokens,
            in_domains=args.in_domains,
            loss_on_unmasked=args.loss_on_unmasked,
            alphas=args.alphas,
            sample_tasks_uniformly=args.sample_tasks_uniformly,
            standardize_depth=args.standardize_depth,
            extra_norm_pix_loss=args.extra_norm_pix_loss,
            fp32_output_adapters=args.fp32_output_adapters.split('-')
        )
        
        if log_writer is not None:
            log_writer.update({**{k: v for k, v in train_stats.items()}, 'epoch': epoch})
        
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, loss_balancer=loss_balancer_without_ddp, epoch=epoch)

        log_stats = {**{k: v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    opts = get_args()
    
    if opts.output_dir: Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(opts)
