# coding=utf-8
# Copyright 2024 QiYuan Inc.
import os
import re
import sys
import time
import math
from typing import Dict
from itertools import chain
from collections import defaultdict

import torch
import bmtrain as bmt

sys.path.append("../../")
from fm9g.arguments import get_args
from fm9g.dragonfly.modeling_dragonfly import Dragonfly
from fm9g.dragonfly.modeling_dragonfly import DragonflyConfig
from fm9g.dragonfly.training_tasks.pretrain_indexed import CudaPrefetcher, MixedIndexedDataset, UnpadBatchedMixedDataset
from fm9g.utils import exporter
from fm9g.utils import logger
from fm9g.utils.exporter import save_every_step_stats
from fm9g.utils.training_stats import num_non_embedding_parameters
from fm9g.utils.training_stats import num_parameters

from opendelta import LoraModel
from opendelta.utils.inspect import inspect_optimizer_statistics


def get_tokenizer(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    return tokenizer


def get_model(args):
    config = DragonflyConfig.from_json_file(args.model_config)
    config.tp = 1 if args.tp_size != 1 else 0  # TODO
    config.pose_prob = args.pose_prob
    config.pose_scaling_factor = args.pose_scaling_factor
    config.rope_scaling_type = args.rope_scaling_type
    config.rope_scaling_factor = args.rope_scaling_factor
    config.orig_max_length = args.orig_max_length

    bmt.print_rank("model config: {}".format(config))
    bmt.print_rank("bmt config: {}".format(bmt.config))

    model = Dragonfly(config)
    if args.load is not None:
        bmt.print_rank("args.load is not None, start to load checkpoints" + args.load)
        exporter.load_model_ckpt(args, model)
    else:
        bmt.print_rank("args.load is None, start to initialize parameters")
        bmt.init_parameters(model)

    if args.delta_tuning:
        
        delta_model = LoraModel(
            backbone_model=model, 
            modified_modules=eval(args.lora_layer), 
            backend='bmt', 
            lora_r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
        delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
        
        if bmt.rank() == 0:
            print("After freeze: ")
            delta_model.log()

    # for n, v in model.named_parameters():
    #     print(n, v.shape, v.requires_grad)
    # exit(0)

    return model


def get_optimizer(args, model):
    scale_lr_group = []
    normal_group = []
    scale_lr_group_name, normal_group_name = [], []
    for n, p in model.named_parameters():
        if n.endswith(".weight") and "layernorm" not in n and "embedding" not in n and "lm_head" not in n:
            scale_lr_group.append(p)
            scale_lr_group_name.append(n)
        else:
            normal_group.append(p)
            normal_group_name.append(n)
    bmt.print_rank(scale_lr_group_name, normal_group_name)
    param_groups = [
        {"params": scale_lr_group, "lr": args.lr / model.config.scale_width},
        {"params": normal_group, "lr": args.lr},
    ]

    if args.offload:
        optimizer = bmt.optim.AdamOffloadOptimizer(param_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    else:
        optimizer = bmt.optim.AdamOptimizer(param_groups, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    if args.load is not None and args.load_grad and (not args.load.endswith(".pt")):
        exporter.load_optimizer_ckpt(args, optimizer)
        bmt.print_rank("optimizer is loaded!")
        
    if args.delta_tuning:
        inspect_optimizer_statistics(optimizer)

    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    from fm9g.training_utils.lr_scheduler import Cosine
    from fm9g.training_utils.lr_scheduler import WarmupStableDrop

    end_iter = args.train_iters
    if 0 < args.warmup_iters < 1:  # 需要支持按固定比例step用来做warmup的
        warmup_iters = int(end_iter * args.warmup_iters)
    else:
        warmup_iters = int(args.warmup_iters)

    if 0 < args.drop_iters < 1:  # 需要支持按固定比例step用来做drop的
        drop_iters = int(end_iter * args.drop_iters)
    else:
        drop_iters = int(args.drop_iters)

    if args.lr_scheduler == "cosine":
        lr_scheduler = Cosine(
            optimizer,
            start_lr=args.lr,
            warmup_iter=warmup_iters,
            end_iter=end_iter,  # 原来是lr_decay_iter
            num_iter=args.start_step,
            #lr_end_restart=args.lr_end_restart,
            #resume_no_optimze=args.resume_no_optimze,
        )
    elif args.lr_scheduler == "warmupstabledrop":
        lr_scheduler = WarmupStableDrop(
            optimizer,
            start_lr=args.lr,
            warmup_iter=warmup_iters,
            end_iter=end_iter,  # 原来是lr_decay_iter
            drop_iter=drop_iters,
            num_iter=args.start_step,
            resume_no_optimze=args.resume_no_optimze,
        )
    return lr_scheduler


def setup_model_and_optimizer(args):
    start = time.time()
    tokenizer = get_tokenizer(args)
    bmt.synchronize()
    logger.info("load tokenizer in {:.2f}s".format(time.time() - start))

    start = time.time()
    model = get_model(args)
    logger.info("load model in {:.2f}s".format(time.time() - start))

    start = time.time()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    logger.info("load lr_scheduler in {:.2f}s".format(time.time() - start))

    return tokenizer, model, optimizer, lr_scheduler


def resume_training(args):
    ckpts = sorted(
        [z for z in chain(*[[os.path.join(x[0], y) for y in x[2]] for x in os.walk(args.save)]) if z.endswith(".pt")],
        reverse=True,
        key=lambda x: (int)(re.search("(\d+).pt", x)[1]),
    )
    # find newest job
    ckpts = sorted(
        ckpts,
        reverse=True,
        key=lambda x: (int)(re.search("job_(\d+)_ckpt", x)[1]),
    )

    if len(ckpts) > 0:
        bmt.print_rank(f"resuming with last checkpoint: {ckpts[0]}")
        args.load = ckpts[0]
        # by default, do not load grad file
        args.load_grad = False
        args.start_step = 0
    else:
        # no ckpts, nothing we can do
        os._exit(1)


def initialize():
    args = get_args(pretrain=True, finetune=True)
    bmt.init_distributed(seed=args.seed, tp_size=args.tp_size)

    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    if args.load is not None:
        if args.load.endswith(".pt"):
            logger.info("You load model ckpt, and choose to completely start the 0 step.")
        elif args.only_load_model == 0:
            if args.start_step == 0:
                log_ckpt = exporter.load_log_ckpt(args)
                if "iteration" in log_ckpt:
                    args.start_step = log_ckpt["iteration"]
                else:
                    args.start_step = (int)(re.findall("(\d+)", args.load)[-1])
                logger.info("Start from step {}".format(args.start_step))
        elif args.only_load_model == 1:
            logger.info("You load model ckpt, and choose to completely start the 0 step.")
        else:
            raise NotImplementedError
    else:
        logger.info("You do not load model")

    return args


def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (
            round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 2),
        )
    torch.cuda.reset_peak_memory_stats()
    return res


def add_mem_time(info, mem_usage, tim_usage):
    torch.cuda.synchronize()
    bmt.synchronize()
    mem_usage[info] = see_memory()
    tim_usage[info] = time.time()
    return mem_usage, tim_usage


def get_task_loss_and_token(loss, task_ids, task_num, targets):
    # task_ids 可能有-1 来代表无效token
    _task_num = task_num + 1
    _task_ids = (task_ids.clone() + 1).to(torch.int64)  # [batch_size, seq_len]
    # gen masks
    _task_mask = torch.zeros((_task_num, *_task_ids.shape), device=_task_ids.device)
    _task_mask.scatter_(0, _task_ids.unsqueeze(0), 1)  # [task_num, batch_size, seq_len]
    _loss_mask = torch.ne(targets, -100).to(torch.int32)
    _mask = _task_mask * _loss_mask.unsqueeze(0)  # [task_num, batch_size, seq_len]
    # calc loss and tokens
    _task_losses = (loss.unsqueeze(0) * _mask).view((_task_num, -1)).sum(dim=-1)[1:]  # [task_num]
    _task_tokens = _mask.view((_task_num, -1)).sum(dim=-1)[1:]  # [task_num]
    # return token-wise avg losses and tokens
    return torch.nan_to_num(_task_losses / _task_tokens, nan=0.0), _task_tokens


class ChunkAve:
    def __init__(self, chunk_size=100):
        self.ave_list = []
        self.chunk_size = chunk_size

    def record(self, time):
        self.ave_list.append(time)
        self.ave_list = self.ave_list[-self.chunk_size :]

    def get(self):
        return sum(self.ave_list) / len(self.ave_list)


def pretrain(
    args,
    tokenizer,
    model: Dragonfly,
    optimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
):
    ave_model_time = ChunkAve(chunk_size=100)
    ave_iter_time = ChunkAve(chunk_size=100)

    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")
    optim_manager = bmt.optim.OptimManager(
        loss_scale=None,
        loss_scale_steps=args.loss_scale_steps,
        loss_scale_factor=2,
        max_loss_scale=args.max_loss_scale,
        min_loss_scale=args.min_loss_scale,
    )
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    start_step = args.start_step

    if args.tensorboard is not None and bmt.rank() == 0:
        import distutils.version  # noqa: F401

        from tensorboardX import SummaryWriter

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    if args.load is not None:
        log_ckpt = exporter.load_log_ckpt(args)
    else:
        log_ckpt = {}
    global_token_pass = log_ckpt.get("global_token_pass", 0.0)
    global_total_task_token = defaultdict(int, log_ckpt.get("global_total_task_token", {}))  # token by task

    global_world_size = bmt.world_size()
    bmt.print_rank("Begin preparing dataset")
    if args.tp_size == 1 or bmt.config["tp_rank"] == 0:
        mixed_indexed_dataset = MixedIndexedDataset(
            cfg_path=args.dataset,
            cfg_json_str=None,
            tokenizer=tokenizer,
            max_length=args.max_length,
            nthreads=args.dataloader_num_threads,
            prefetch_slice=args.dataloader_prefetch,
            weight_by_size=True,
        )

        if args.load is not None and args.only_load_model == 0 and args.load_dataloader_ckpt == 1 and (not args.load.endswith(".pt")):
            exporter.load_dataloader_ckpt(args, mixed_indexed_dataset)

        batched_dataset = UnpadBatchedMixedDataset(mixed_indexed_dataset, args.batch_size, args.max_length)
        dataloader = torch.utils.data.DataLoader(
            batched_dataset,
            batch_size=None,
            collate_fn=lambda x: x,
            num_workers=args.dataloader_num_workers,
            prefetch_factor=args.dataloader_prefetch_factor,
        )
    else:

        def dummy_generator():
            while True:
                yield None

        mixed_indexed_dataset = dummy_generator()
        dataloader = mixed_indexed_dataset

    DataIterator = CudaPrefetcher(dataloader, tp_size=args.tp_size, tp_rank=bmt.config["tp_rank"])

    bmt.print_rank("Preparing dataset done.")

    # inspect at init
    model_inspect = bmt.inspect.inspect_model(model, "*")
    bmt.print_rank(bmt.inspect.format_summary(model_inspect))

    try:
        mem_usage, tim_usage = {}, {}
        mem_usage, tim_usage = add_mem_time("before_log", mem_usage, tim_usage)

        for iteration, data in enumerate(DataIterator, start=start_step + 1):
            if args.tp_size == 1 or bmt.config["tp_rank"] == 0:
                mixed_indexed_dataset.update_states(data["task_ids"], data["indexes"])

            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            logits = model(
                input=data["inputs"],
                cu_seqlens=data["cu_seqlens"],
                max_seqlen=data["max_seqlen"],
                position_ids=data["position_ids"],
            )

            # chunk targets and task_ids
            data["targets"] = (
                data["targets"]
                .view(-1)
                .chunk(bmt.config["tp_size"])[bmt.config["tp_rank"]]
                .view(data["targets"].shape[0], -1)
            )
            data["task_ids"] = (
                data["task_ids"]
                .view(-1)
                .chunk(bmt.config["tp_size"])[bmt.config["tp_rank"]]
                .view(data["task_ids"].shape[0], -1)
            )

            _target = data["targets"].view(-1)
            non_reduced_loss = loss_func(logits.view(-1, logits.size(-1)), _target)
            _w = (_target != -100).int()
            loss = non_reduced_loss.sum() / _w.sum().float()

            global_loss = bmt.sum_loss(loss).item()
            mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

            optim_manager.backward(loss)
            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

            if iteration % args.grad_accum == 0 or iteration == args.train_iters:
                grad_accum_init_time = tim_usage["init"]

                grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type=2)
                optim_manager.step()
                optim_manager.zero_grad()
                mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)
                model_time = tim_usage["optim"] - grad_accum_init_time
                ave_model_time.record(model_time)
            else:
                # dummy optim step
                grad_norm = torch.Tensor([0.0]).cuda()
                tim_usage["optim"] = tim_usage["backward"]
                mem_usage["optim"] = mem_usage["backward"]

            with torch.no_grad():
                task_num = len(data["task_names"])
                task_loss, task_token = get_task_loss_and_token(
                    non_reduced_loss, data["task_ids"], task_num, data["targets"]
                )
                task_loss_map: Dict[str, float] = {}
                gatherd_task_loss_map = bmt.distributed.all_gather(task_loss)
                gatherd_task_token_map = bmt.distributed.all_gather(task_token)
                gatherd_task_loss_token_map = gatherd_task_loss_map * gatherd_task_token_map
                sum_task_loss = gatherd_task_loss_token_map.sum(dim=0)
                tot_task_token = gatherd_task_token_map.sum(dim=0)
                ave_task_loss = sum_task_loss / tot_task_token
                for i in range(task_num):
                    task_loss_map[data["task_names"][i]] = ave_task_loss[i].item()
                    global_total_task_token[data["task_names"][i]] += tot_task_token[i].item()

            local_total_rate = torch.Tensor([data["lengths"].float().mean() / args.max_length]).cuda()
            local_total_rate = bmt.sum_loss(local_total_rate).item()
            global_token_pass += (
                (global_world_size // args.tp_size) * local_total_rate * args.max_length * args.batch_size
            )

            bmt.print_rank(
                "=========================================" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            )
            last_before_log_time = tim_usage["before_log"]
            mem_usage, tim_usage = add_mem_time("before_log", mem_usage, tim_usage)

            iter_time = tim_usage["before_log"] - last_before_log_time

            ave_iter_time.record(iter_time)

            train_info = {
                "time": iter_time,
                "iteration": iteration,
                "loss": global_loss,
                "lr": lr_scheduler.current_lr,
                "token_max": local_total_rate,
                "token_pass": global_token_pass,
                "throughout": args.max_length * args.batch_size * local_total_rate / ave_iter_time.get() / args.tp_size,
                "grad_norm": grad_norm.item(),
                "mask_max": ((data["targets"] >= 0).sum(-1).float().mean() / args.max_length).item(),
                "task_loss": task_loss_map,
                "total_task_token": global_total_task_token,
            }
            global_token_pass_str = convert_to_k_and_b(global_token_pass)

            bmt.print_rank(
                (
                    "| Iter: {iteration:6d} | loss: {loss:.4f} | lr: {lr:.4e} | model_time: {model_time:.2f} | iter_time: {iter_time:.2f}| chunk_ave_time: {chunk_ave_time:.2f}"
                    + " token/max: {tokenrate:.4f} | mask/max: {maskrate:.4f} | grad_norm: {grad_norm:.4f} | global_token_pass (B):"
                    + "{global_token_pass} | mem_usage {mem_usage} | "
                ).format(
                    iteration=iteration,
                    loss=global_loss,
                    lr=lr_scheduler.current_lr,
                    model_time=model_time,
                    iter_time=iter_time,
                    chunk_ave_time=ave_iter_time.get(),
                    tokenrate=data["lengths"].float().mean() / args.max_length / args.batch_size,
                    maskrate=(data["targets"] >= 0).sum(-1).float().mean() / args.max_length / args.batch_size,
                    grad_norm=grad_norm.item(),
                    global_token_pass=global_token_pass_str,
                    mem_usage=max([value for key, value in mem_usage.items()]),
                )
            )

            bmt.print_rank(
                "task_loss:\t| "
                + " | ".join(["{}: {:.4f}".format(task_name, loss) for task_name, loss in task_loss_map.items()])
                + " |"
            )

            if iteration % 10 == 0:
                bmt.print_rank(
                    "task_tokens (B):\t| "
                    + " | ".join(
                        [
                            "{}: {:.4f}".format(task_name, task_token / 10**9)
                            for task_name, task_token in global_total_task_token.items()
                        ]
                    )
                    + " |"
                )

            if iteration % args.inspect_iters == 0:
                model_inspect = bmt.inspect.inspect_model(model, "*")
                bmt.print_rank(bmt.inspect.format_summary(model_inspect))

            if args.log_dir is not None and bmt.rank() == 0:
                if args.save is not None:
                    save_every_step_stats(train_info, args.save)

            if args.tensorboard is not None and bmt.rank() == 0:
                writer.add_scalar("Loss/train", global_loss, iteration)
                writer.add_scalar("Optimizer/lr", lr_scheduler.current_lr, iteration)
                writer.add_scalar("Optimizer/scale", optim_manager.loss_scale, iteration)
                writer.add_scalar("Optimizer/grad_norm", grad_norm.item(), iteration)
                for task_name, loss in task_loss_map.items():
                    if not math.isnan(loss):
                        writer.add_scalar("Loss/train/{}".format(task_name), loss, iteration)

            # -------- save file. If need to backup by Klara platform, use export.xx_save --------
            log_ckpt = {
                "global_total_task_token": global_total_task_token,
                "global_token_pass": global_token_pass,
                "iteration": iteration,
            }

            if args.save is not None and iteration % args.save_iters == 0:
                exporter.export(
                    model,
                    mixed_indexed_dataset,
                    tokenizer,
                    optimizer,
                    iteration,
                    args,
                    log_ckpt=log_ckpt,
                    final_save=False,
                )

            if iteration == args.train_iters and args.stop_when_end == 1:
                break

    except Exception as e:
        print(f"train loop err: {e}")
        raise e
    finally:
        pass
    
    exporter.export(model, mixed_indexed_dataset, tokenizer, optimizer, -1, args, final_save=False)


def convert_to_k_and_b(number):
    if number >= 1e9:  # 大于或等于10亿
        b_number = number / 1e9
        return f"{b_number:.2f}B"
    elif number >= 1e6:  # 大于或等于1百万
        k_number = number / 1e6
        return f"{k_number:.2f}M"
    elif number >= 1e3:
        k_number = number / 1e3
        return f"{k_number:.2f}K"
    else:
        return str(number)


def main():
    args = initialize()
    bmt.synchronize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    bmt.print_rank("finish loading")
    # bmt.print_rank(
    #     "Number of parameter {}, Number of non-e parameter {}".format(
    #         num_parameters(model), num_non_embedding_parameters(model)
    #     )
    # )
    bmt.print_rank("args: {}".format(args))

    pretrain(args, tokenizer, model, optimizer, lr_scheduler)


if __name__ == "__main__":
    main()
