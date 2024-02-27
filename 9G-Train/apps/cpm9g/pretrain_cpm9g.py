import inspect
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import bmtrain as bmt
import torch

sys.path.insert(0, "/home/wangshuo1/code/9G-Train")
from cpm.arguments import get_args
from cpm.cpm9g.models import CPM9G
from cpm.cpm9g.models import CPM9GConfig
from cpm.cpm9g.tokenizers import CPM9GTokenizer
from cpm.cpm9g.training_tasks import MixedDataset
from cpm.utils import allgather_objects
from cpm.utils import exporter
from cpm.utils import logger
from cpm.utils import LogManager


def get_tokenizer(args):
    tokenizer = CPM9GTokenizer(path=args.vocab)
    return tokenizer


def get_model(args):
    config = CPM9GConfig.from_json_file(args.model_config)
    config.tp = 1 if args.tp != 1 else 0
    if args.flash == "none":
        config.use_flash_attn = False
    else:
        config.use_flash_attn = True
        if args.flash == "1d":
            config.flash_attn_mask_shape = "1d"
        else:
            config.flash_attn_mask_shape = "2d"
            if args.flash == "triton":
                config.flash_impl = "triton"
            elif args.flash == "cuda":
                config.flash_impl = "cuda"
    model = CPM9G(config)
    if args.load is not None:
        bmt.print_rank("args.load is not None, start to load checkpoints" + args.load)
        bmt.load(model, args.load)
    else:
        bmt.print_rank("args.load is None, start to initialize parameters")
        bmt.init_parameters(model)
    return model


def get_optimizer(args, model):
    for name, para in model.named_parameters():
        # if not ('input_embedding' in name or 'lm_head' in name):
        #     para.requires_grad_(False)
        bmt.print_rank(name, para.requires_grad)
    if args.offload:
        optimizer = bmt.optim.AdamOffloadOptimizer(
            model.parameters(), betas=(0.9, 0.95), weight_decay=args.weight_decay
        )
    else:
        optimizer = bmt.optim.AdamOptimizer(model.parameters(), betas=(0.9, 0.95), weight_decay=args.weight_decay)
    if args.load is not None and args.load_grad:
        start = time.time()
        print(
            sum([1 if re.search(r"-{}.rank-\d+.opt".format(args.start_step), i) else 0 for i in os.listdir(args.save)])
        )
        if (
            sum([1 if re.search(r"-{}.rank-\d+.opt".format(args.start_step), i) else 0 for i in os.listdir(args.save)])
            == bmt.world_size()
        ):
            file_name = os.path.join(
                args.save,
                args.save_name + "-{}.rank-{}.opt".format(args.start_step, bmt.rank()),
            )
            print(file_name)
            if os.path.exists(file_name):
                print("start to load grad ckpt {}".format(file_name))
                states = torch.load(file_name)
                optimizer.load_state_dict(states)
        logger.info("load grad in {:.2f}s".format(time.time() - start))
    return optimizer


class Cosine(bmt.lr_scheduler.WarmupLRScheduler):
    r"""
    After a warmup period during which learning rate increases linearly between 0 and the start_lr,
    The decay period performs :math:`\text{lr}=\text{start_lr}\times \dfrac{1+\cos \left( \pi \cdot \dfrac{\text{num_iter}-\text{warmup_iter}}{\text{end_iter}-\text{warmup_iter}}\right)}{2}`
    """

    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr * num_iter / self.warmup_iter

    def get_lr_decay(self, num_iter) -> float:
        progress = (num_iter - self.warmup_iter) / max(1, (self.end_iter - self.warmup_iter))
        return max(self.start_lr * 0.1, self.start_lr * (0.1 + 0.45 * (1.0 + math.cos(progress * math.pi))))


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    # lr_scheduler = bmt.lr_scheduler.Noam(
    lr_scheduler = Cosine(
        optimizer,
        start_lr=args.lr,
        warmup_iter=args.warmup_iters,
        end_iter=args.lr_decay_iters,
        num_iter=args.start_step,
    )
    return lr_scheduler


def setup_model_and_optimizer(args):

    start = time.time()
    model = get_model(args)
    logger.info("load model in {:.2f}s".format(time.time() - start))

    start = time.time()
    tokenizer = get_tokenizer(args)
    bmt.synchronize()
    logger.info("load tokenizer in {:.2f}s".format(time.time() - start))

    start = time.time()
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    logger.info("load lr_scheduler in {:.2f}s".format(time.time() - start))

    return tokenizer, model, optimizer, lr_scheduler


def initialize():
    args = get_args(pretrain=True)
    bmt.init_distributed(seed=args.seed, zero_level=3)
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    if args.load is not None:
        if args.start_step == 0:
            args.start_step = (int)(re.search("(\d+).pt", args.load)[1])
    return args


def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (
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


class LossSpikeDetector:
    def __init__(self, log_path: str) -> None:
        self._last_loss: Dict[str, float] = {}
        self._last_data: List[Any] = [None]
        self._log_path = log_path

    def update_data(self, data: Any):
        self._last_data.append(data)
        if len(self._last_data) > 2:
            self._last_data = self._last_data[-2:]

    def update_loss(self, iteration: int, loss_map: Dict[str, float]):
        loss_spike_result = []
        for task, loss in loss_map.items():
            if task in self._last_loss:
                if loss > self._last_loss[task] * 3:
                    # loss spike!
                    loss_spike_result.append(
                        {
                            "prev": self._last_loss[task],
                            "curr": loss,
                            "task": task,
                        }
                    )
            self._last_loss[task] = float(loss)
        if len(loss_spike_result) > 0:
            self._write_log(iteration, self._last_data[-1], loss_spike_result)

    def _write_log(self, iteration: int, data: Any, result: List[Dict[str, Any]]):
        while True:
            try:
                with open(self._log_path, "a", encoding="utf-8") as fp:
                    fp.write("=" * 20)
                    fp.write("\nloss spike at {}\n".format(iteration))
                    fp.write("{}\n".format(json.dumps(result, indent=4, ensure_ascii=False)))
                    fp.write("data: \n")
                    for d in data:
                        fp.write("{}\n".format(json.dumps(d, indent=4, ensure_ascii=False)))
                    fp.write("\n\n")
                    break
            except Exception as e:
                print("cannot output log to the file {}", self._log_path)


def pretrain(
    args,
    tokenizer: CPM9GTokenizer,
    model: CPM9G,
    optimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
):
    average_time = bmt.utils.AverageRecorder()
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    optim_manager = bmt.optim.OptimManager(
        loss_scale=None if args.bf16 else args.loss_scale,
        loss_scale_steps=args.loss_scale_steps,
        loss_scale_factor=2,
        max_loss_scale=args.max_loss_scale,
        min_loss_scale=args.min_loss_scale,
    )
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    start_step = args.start_step
    lsd = LossSpikeDetector("./log/debug/spile.%d.log" % bmt.rank())

    if args.tensorboard is not None and bmt.rank() == 0:
        import distutils.version  # noqa: F401

        from tensorboardX import SummaryWriter

        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    if args.log_dir is not None and bmt.rank() == 0:
        log_mgr = LogManager(args.log_dir)

    global_token_pass = 0.0
    global_world_size = bmt.world_size()
    dataloader = MixedDataset(args.dataset, args.batch_size, args.max_length, tokenizer, unpad=(args.flash == "cuda"))

    if args.load is not None:
        dataset_states_path = args.load.replace(".pt", ".data")
        if os.path.exists(dataset_states_path):
            start = time.time()
            bmt.print_rank("start to load data ckpt")
            dataset_states = torch.load(dataset_states_path)
            logger.info("load data ckpt in {:.2f}s".format(time.time() - start))

            start = time.time()
            missing = dataloader.load_state_dict(dataset_states)
            logger.info("load state dict in {:.2f}s".format(time.time() - start))
            if len(missing) > 0:
                bmt.print_rank("Missing keys when loading dataset states: ", missing)
        else:
            bmt.print_rank("cannot find data ckpt {}".format(dataset_states_path))

    dataloader.start()
    bmt.print_rank("finish dataset start")
    try:
        total = 0
        hash = {}
        for iteration, data in enumerate(dataloader):
            iteration = iteration + start_step + 1
            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            targets = torch.from_numpy(data["target"]).cuda().to(torch.int32)
            task_ids = torch.from_numpy(data["task_ids"]).cuda().to(torch.int32)
            task_names = data["task_names"]
            lsd.update_data(data["raw_data"])
            if args.flash == "cuda":
                cu_seqlens = torch.from_numpy(data["cu_seqlens"]).cuda().to(torch.int32)
                max_seqlen = data["max_seqlen"]
                position_ids = torch.from_numpy(data["position_ids"]).cuda().to(torch.int32)
            else:
                input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
                input_context = torch.zeros_like(input_ids).cuda().bool()
                input_span = torch.from_numpy(data["spans"]).cuda().to(torch.int32)

            # ===========
            optim_manager.zero_grad()
            # torch.cuda.empty_cache()
            mem_usage = {}
            tim_usage = {}
            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

            # bmt.print_rank(torch.cuda.max_memory_allocated())
            # ===========
            if args.flash == "cuda":
                logits, _ = model(
                    input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    position_ids=position_ids,
                )
            else:
                logits, _ = model(
                    input_ids,
                    input_length,
                    input_context,
                    input_span,
                )
            mem_usage, tim_usage = add_mem_time("forward_1", mem_usage, tim_usage)
            loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
            global_loss = bmt.sum_loss(loss).item()
            mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

            # bmt.print_rank(torch.cuda.max_memory_allocated())
            # ===========
            optim_manager.backward(loss)
            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

            # bmt.print_rank(torch.cuda.max_memory_allocated())
            # ===========
            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type=2)
            optim_manager.step()
            mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)
            # bmt.print_rank(torch.cuda.max_memory_allocated())

            # ==========
            iter_time = tim_usage["optim"] - tim_usage["init"]
            average_time.record(iter_time)

            with torch.no_grad():
                task_num = len(task_names)
                targets_tmp = targets.expand(task_num, -1, -1)
                task = torch.arange(task_num, dtype=torch.int32, device="cuda")[:, None, None]
                targets_tmp = torch.where(
                    task_ids == task,
                    targets_tmp,
                    torch.scalar_tensor(-100, dtype=torch.int32, device="cuda"),
                )

                task_loss_map: Dict[str, float] = {}
                task_loss_tot: Dict[str, float] = {}
                for i in range(task_num):
                    task_loss_map[task_names[i]] = loss_func(
                        logits.view(-1, logits.size(-1)), targets_tmp[i, :].view(-1)
                    ).item()
                    task_loss_tot[task_names[i]] = (targets_tmp[i, :].view(-1) >= 0).sum().float().item()
                gatherd_task_loss_map: List[Dict[str, float]] = allgather_objects(task_loss_map)
                gatherd_task_loss_tot: List[Dict[str, float]] = allgather_objects(task_loss_tot)

                global_task_loss_map: Dict[str, Union[List[float], float]] = {}
                global_task_loss_tot: Dict[str, Union[List[float], float]] = {}

                for idx, local_task_loss_map in enumerate(gatherd_task_loss_map):
                    for task_name, task_loss in local_task_loss_map.items():
                        if task_name not in global_task_loss_map:
                            global_task_loss_map[task_name] = []
                        global_task_loss_map[task_name].append(task_loss)
                    for task_name, task_tot in gatherd_task_loss_tot[idx].items():
                        if task_name not in global_task_loss_tot:
                            global_task_loss_tot[task_name] = []
                        global_task_loss_tot[task_name].append(task_tot)

                task_loss_map = {}
                for task_name in sorted(list(global_task_loss_map.keys())):
                    avg_loss = 0.0
                    sum_token = sum(global_task_loss_tot[task_name])
                    for loss, token in zip(global_task_loss_map[task_name], global_task_loss_tot[task_name]):
                        avg_loss += loss * token / sum_token
                    task_loss_map[task_name] = avg_loss

            local_total_rate = torch.Tensor([input_length.float().mean() / args.max_length]).cuda()
            local_total_rate = bmt.sum_loss(local_total_rate).item()
            global_token_pass += global_world_size * local_total_rate * args.max_length * args.batch_size
            avg_time = average_time.value
            lsd.update_loss(iteration, task_loss_map)

            for task_id in data["task_ids"]:
                for task in task_id:
                    if task != -1:
                        if not data["task_names"][task] in hash:
                            hash[data["task_names"][task]] = 0
                        hash[data["task_names"][task]] += 1.0
                        total += 1.0

            gathered_hash = allgather_objects(hash)
            sum_total = sum(allgather_objects(total))

            final_hash = defaultdict(int)
            for local_hash in gathered_hash:
                for task, num in local_hash.items():
                    final_hash[task] += num

            # for i in final_hash:
            #     bmt.print_rank(i, final_hash[i] / sum_total)
            # bmt.print_rank("=========================================")

            train_info = {
                "time": tim_usage["init"],
                "iteration": iteration,
                "loss": global_loss,
                "lr": lr_scheduler.current_lr,
                "lr_scale": int(optim_manager.loss_scale),
                "time_usage": tim_usage,
                "mem_usage": mem_usage,
                "avg_time": avg_time,
                "token_max": local_total_rate,
                "token_pass": global_token_pass,
                "throughout": args.max_length * args.batch_size * local_total_rate / avg_time,
                "grad_norm": grad_norm.item(),
                "mask_max": ((targets >= 0).sum(-1).float().mean() / args.max_length).item(),
                "num_gpus": global_world_size,
                "task_loss": task_loss_map,
            }
            # bmt.print_rank(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            bmt.print_rank(
                (
                    "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | avg_time: {:.4f}; cur_time:{:.4f}={:.4f}+{:.4f} |"
                    + " token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f} | mem: {:.2f} |"
                ).format(
                    iteration,
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optim_manager.loss_scale),
                    iter_time,
                    tim_usage["optim"] - tim_usage["init"],
                    tim_usage["backward"] - tim_usage["init"],
                    tim_usage["optim"] - tim_usage["backward"],
                    input_length.float().mean() / args.max_length / (args.batch_size if args.flash == "cuda" else 1),
                    (targets >= 0).sum(-1).float().mean()
                    / args.max_length
                    / (args.batch_size if args.flash == "cuda" else 1),
                    grad_norm,
                    max(mem_usage["forward"][1], mem_usage["backward"][1]),
                )
            )

            bmt.print_rank(
                "| "
                + " | ".join(["{}: {:.4f}".format(task_name, loss) for task_name, loss in task_loss_map.items()])
                + " |"
            )

            if iteration % args.inspect_iters == 0:
                model_inspect = bmt.inspect.inspect_model(model, "*")
                bmt.print_rank(bmt.inspect.format_summary(model_inspect))
                train_info["model_inspect"] = model_inspect

            if args.log_dir is not None and bmt.rank() == 0:
                log_mgr.write(**train_info)
            if args.tensorboard is not None and bmt.rank() == 0:
                writer.add_scalar("Loss/train", global_loss, iteration)
                writer.add_scalar("Optimizer/lr", lr_scheduler.current_lr, iteration)
                writer.add_scalar("Optimizer/scale", optim_manager.loss_scale, iteration)
                writer.add_scalar("Optimizer/grad_norm", grad_norm.item(), iteration)
                for task_name, loss in task_loss_map.items():
                    writer.add_scalar("Loss/train/{}".format(task_name), loss, iteration)

            # -------- save file. If need to backup by Klara platform, use export.xx_save --------
            if args.save is not None and iteration % args.save_iters == 0:
                exporter.export(model, dataloader, optimizer, iteration, args, final_save=False)
            
            if iteration >= args.train_iters:
                break

    except Exception as e:
        print(f"train loop err: {e}")
        raise e
    finally:
        dataloader.close()
    exporter.export(model, dataloader, optimizer, -1, args, final_save=False)


def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    bmt.print_rank("finish loading")
    pretrain(args, tokenizer, model, optimizer, lr_scheduler)


if __name__ == "__main__":
    main()
