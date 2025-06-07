import math
import torch
import utils

from einops import rearrange
from utils import create_model
from typing import Dict, Iterable, List

def get_model(args, DOMAIN_CONF):

    print(f"Creating model: {args.model} for inputs {args.in_domains} and outputs {args.out_domains}")

    input_adapters = {
        domain: DOMAIN_CONF[domain]['input_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
        )
        for domain in args.in_domains
    }

    output_adapters = {
        domain: DOMAIN_CONF[domain]['output_adapter'](
            stride_level=DOMAIN_CONF[domain]['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task=domain,
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn
        )
        for domain in args.out_domains
    }

    # Add normalized pixel output adapter if specified
    if args.extra_norm_pix_loss:
        output_adapters['norm_rgb'] = DOMAIN_CONF['rgb']['output_adapter'](
            stride_level=DOMAIN_CONF['rgb']['stride_level'],
            patch_size_full=args.patch_size,
            dim_tokens=args.decoder_dim,
            depth=args.decoder_depth,
            num_heads=args.decoder_num_heads,
            use_task_queries=args.decoder_use_task_queries,
            task='rgb',
            context_tasks=list(args.in_domains),
            use_xattn=args.decoder_use_xattn)

    model = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path
    )

    return model

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, tasks_loss_fn: Dict[str, torch.nn.Module],
                    loss_balancer: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = None, max_skip_norm: float = None,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_encoded_tokens: int = 196, in_domains: List[str] = [] , loss_on_unmasked: bool = True,
                    alphas: float = 1.0, sample_tasks_uniformly: bool = False, standardize_depth: bool = True,
                    extra_norm_pix_loss: bool = False, fp32_output_adapters: List[str] = [], clip_loss: bool = False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, (x, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        tasks_dict = {
            task: tensor.to(device, non_blocking=True)
            for task, tensor in x.items()
        }

        # Truncated depth standardization
        if standardize_depth and 'depth' in tasks_dict:
            # Flatten depth and remove bottom and top 10% of values
            trunc_depth = torch.sort(rearrange(tasks_dict['depth'], 'b c h w -> b (c h w)'), dim=1)[0]
            trunc_depth = trunc_depth[:,int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
            tasks_dict['depth'] = (tasks_dict['depth'] - trunc_depth.mean(dim=1)[:,None,None,None]) / torch.sqrt(trunc_depth.var(dim=1)[:,None,None,None] + 1e-6)

        input_dict = {
            task: tensor
            for task, tensor in tasks_dict.items()
            if task in in_domains or task in ['input_ids', 'attention_mask']
        }

        with torch.cuda.amp.autocast():
            preds, masks = model(input_dict, 
                num_encoded_tokens=num_encoded_tokens, 
                alphas=alphas, 
                sample_tasks_uniformly=sample_tasks_uniformly,
                fp32_output_adapters=fp32_output_adapters)

            if extra_norm_pix_loss:
                tasks_dict['norm_rgb'] = tasks_dict['rgb']
                masks['norm_rgb'] = masks.get('rgb', None)

            task_losses = {}
            
            for task in preds:
                target = tasks_dict[task]
                    
                if loss_on_unmasked:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target)
                else:
                    task_losses[task] = tasks_loss_fn[task](preds[task].float(), target, mask=masks.get(task, None))
 
            weighted_task_losses = loss_balancer(task_losses)
            loss = sum(weighted_task_losses.values())

        loss_value = sum(task_losses.values()).item()
        task_loss_values = {f'{task}_loss': l.item() for task, l in task_losses.items()}
        weighted_task_loss_values = {f'{task}_loss_weighted': l.item() for task, l in weighted_task_losses.items()}

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**task_loss_values)
        metric_logger.update(loss_scale=loss_scale_value)
        
        min_lr = 10.
        max_lr = 0.
        
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.update(task_loss_values)
            log_writer.update(weighted_task_loss_values)
            log_writer.set_step()

        if lr_scheduler is not None: lr_scheduler.step_update(start_steps + step)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    print("Averaged stats:", metric_logger)
    
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}

