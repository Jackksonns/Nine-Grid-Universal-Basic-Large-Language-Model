# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import json
import math
import os
import re
import sys
import time
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
from cpm.cpm9g.training_tasks import FinetuneDataset
from cpm.utils import allgather_objects
from cpm.utils import logger
import shutil

import opendelta as od
from opendelta import LoraModel, AdapterModel, CompacterModel, LowRankAdapterModel, BitFitModel, ParallelAdapterModel
from opendelta.utils.inspect import inspect_optimizer_statistics
from bigmodelvis import Visualization


def get_tokenizer(args):
    tokenizer = CPM9GTokenizer(path=args.vocab)
    return tokenizer


def get_model(args):
    config = CPM9GConfig.from_json_file(args.model_config)
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
        bmt.init_parameters(model)
        bmt.synchronize()
        bmt.print_rank("args.load is not None, start to load checkpoints" + args.load)
        bmt.load(model, args.load, strict=False)

        model_inspect = bmt.inspect.inspect_model(model, "*")
        bmt.print_rank(bmt.inspect.format_summary(model_inspect))
    else:
        bmt.print_rank("args.load is None, start to initialize parameters")
        bmt.init_parameters(model)

    if args.delta_type != None:
        from bigmodelvis import Visualization #0813
        if bmt.rank() == 0:
            Visualization(model).structure_graph()
            print("finetuned layers: ")
            print(args.lora_layer)
        
        if args.delta_type == "lora":
            delta_model = LoraModel(backbone_model=model, modified_modules=args.lora_layer, backend='bmt', lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

        if bmt.rank() == 0:
            print("Before freeze: ")
            delta_model.log()

        delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)

        if bmt.rank() == 0:
            print("After freeze: ")
            delta_model.log()

    return model


def get_optimizer(args, model):
    if args.offload:
        optimizer = bmt.optim.AdamOffloadOptimizer(
            model.parameters(), betas=(0.9, 0.95), weight_decay=args.weight_decay
        )
    else:
        optimizer = bmt.optim.AdamOptimizer(model.parameters(), betas=(0.9, 0.95), weight_decay=args.weight_decay)
    if args.load is not None and args.load_grad:
        start = time.time()
        print(
            sum(
                [
                    1
                    if i.find(".opt") != -1 and i.find("-{}.rank".format(args.start_step % (args.save_iters * 5))) != -1
                    else 0
                    for i in os.listdir(args.save)
                ]
            )
        )
        if (
            sum(
                [
                    1
                    if i.find(".opt") != -1 and i.find("-{}.rank".format(args.start_step % (args.save_iters * 5))) != -1
                    else 0
                    for i in os.listdir(args.save)
                ]
            )
            == bmt.world_size()
        ):
            file_name = os.path.join(
                args.save,
                args.save_name + "-{}.rank-{}.opt".format(args.start_step % (args.save_iters * 5), bmt.rank()),
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

    if args.delta_type != None:
        inspect_optimizer_statistics(optimizer)

    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    logger.info("load lr_scheduler in {:.2f}s".format(time.time() - start))

    return tokenizer, model, optimizer, lr_scheduler


def initialize():
    args = get_args(finetune=True)

    # hack
    if "checkpointing" in inspect.signature(bmt.init_distributed).parameters:
        bmt.init_distributed(checkpointing=False, seed=args.seed)
    else:
        bmt.init_distributed(seed=args.seed)
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    # if args.load is not None:
    #     if args.start_step == 0:
    #         args.start_step = (int)(re.search("(\d+).pt", args.load)[1])
    return args


def see_memory(detail=False):
    if detail:
        res = torch.cuda.memory_summary()
    else:
        res = (
            round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024), 2),
            round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2),
        )
    torch.cuda.reset_peak_memory_stats()
    return res


def add_mem_time(info, mem_usage, tim_usage):
    torch.cuda.synchronize()
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
        return
        with open(self._log_path, "a", encoding="utf-8") as fp:
            fp.write("=" * 20)
            fp.write("\nloss spike at {}\n".format(iteration))
            fp.write("{}\n".format(json.dumps(result, indent=4, ensure_ascii=False)))
            fp.write("data: \n")
            for d in data:
                fp.write("{}\n".format(json.dumps(d, indent=4, ensure_ascii=False)))
            fp.write("\n\n")


def finetune(
    args,
    bin_file: str,
    tokenizer: CPM9GTokenizer,
    model: CPM9G,
    optimizer,
    lr_scheduler: bmt.lr_scheduler.WarmupLRScheduler,
):
    average_time = bmt.utils.AverageRecorder()
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
    optim_manager = bmt.optim.OptimManager(loss_scale=args.loss_scale, loss_scale_steps=args.loss_scale_steps, loss_scale_factor=2, max_loss_scale=args.max_loss_scale, min_loss_scale=args.min_loss_scale,)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    if args.tensorboard is not None and bmt.rank() == 0:
        import distutils.version  # noqa: F401
        from tensorboardX import SummaryWriter
        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard)
        writer = SummaryWriter(log_dir=args.tensorboard)

    global_token_pass = 0.0
    global_world_size = bmt.world_size()

    for epoch in range(args.epoch):
        epoch = epoch + 1
        last_data = None
        dataloader = FinetuneDataset(
            bin_file, args.batch_size, args.max_length, tokenizer, unpad=(args.flash == "cuda"), task_name="task", drop_last=True
        )
        for iteration, data in enumerate(dataloader):
            iteration = iteration + 1
            skip_this_batch = False
            if data is None:
                if last_data is None:
                    raise RuntimeError(
                        "Dataset is too small, please use a smaller batch size or sequence length!"
                    )
                data = last_data  # use last data
                skip_this_batch = True
            else:
                last_data = data

            assert data["inputs"].shape[0] == args.batch_size
            input_ids = torch.from_numpy(data["inputs"]).cuda().to(torch.int32)
            input_length = torch.from_numpy(data["length"]).cuda().to(torch.int32)
            targets = torch.from_numpy(data["target"]).cuda().long()
            # bmt.print_rank(input_ids[0].tolist())
            # bmt.print_rank(targets[0].tolist())
            # bmt.print_rank(data["spans"].tolist())
            # bmt.print_rank(tokenizer.decode(input_ids[0]))
            # bmt.print_rank(tokenizer.path)
            # bmt.synchronize()
            # exit()
            task_ids = torch.from_numpy(data["task_ids"]).cuda().to(torch.int32)
            task_names = data["task_names"]
            if args.flash == "cuda":
                cu_seqlens = torch.from_numpy(data["cu_seqlens"]).cuda().to(torch.int32)
                max_seqlen = data["max_seqlen"]
                position_ids = torch.from_numpy(data["position_ids"]).cuda().to(torch.int32)
            else:
                input_context = torch.zeros_like(input_ids).cuda().bool()
                input_span = torch.from_numpy(data["spans"]).cuda().to(torch.int32)

            # ===========
            optim_manager.zero_grad()
            # torch.cuda.empty_cache()
            mem_usage = {}
            tim_usage = {}
            mem_usage, tim_usage = add_mem_time("init", mem_usage, tim_usage)

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
            if skip_this_batch:
                loss = loss * 0
            mem_usage, tim_usage = add_mem_time("forward", mem_usage, tim_usage)

            # ===========
            optim_manager.backward(loss)
            mem_usage, tim_usage = add_mem_time("backward", mem_usage, tim_usage)

            # ===========
            grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type=2)
            optim_manager.step()
            mem_usage, tim_usage = add_mem_time("optim", mem_usage, tim_usage)

            # ==========
            iter_time = tim_usage["optim"] - tim_usage["init"]
            average_time.record(iter_time)

            with torch.no_grad():
                task_num = len(task_names)
                targets_tmp = targets.expand(task_num, -1, -1)
                task = torch.arange(task_num, dtype=torch.long, device="cuda")[:, None, None]
                targets_tmp = torch.where(
                    task_ids == task,
                    targets_tmp,
                    torch.scalar_tensor(-100, dtype=torch.long, device="cuda"),
                )

                task_loss_map: Dict[str, float] = {}
                if not skip_this_batch:
                    for i in range(task_num):
                        task_loss = loss_func(logits.view(-1, logits.size(-1)), targets_tmp[i, :].view(-1))
                        task_loss_map[task_names[i]] = task_loss.item()
                gatherd_task_loss_map: List[Dict[str, float]] = allgather_objects(task_loss_map)

                global_task_loss_map: Dict[str, Union[List[float], float]] = {}
                for local_task_loss_map in gatherd_task_loss_map:
                    for task_name, task_loss in local_task_loss_map.items():
                        if task_name not in global_task_loss_map:
                            global_task_loss_map[task_name] = []
                        global_task_loss_map[task_name].append(task_loss)

                task_loss_map = {}
                for task_name in sorted(list(global_task_loss_map.keys())):
                    avg_loss = sum(global_task_loss_map[task_name]) / len(global_task_loss_map[task_name])
                    task_loss_map[task_name] = avg_loss

            local_total_rate = torch.Tensor([input_length.float().mean() / args.max_length]).cuda()
            local_total_rate = bmt.sum_loss(local_total_rate).item()
            global_token_pass += global_world_size * local_total_rate * args.max_length * args.batch_size
            avg_time = average_time.value

            train_info = {
                "time": tim_usage["init"],
                "epoch": epoch,
                "iteration": iteration,
                "loss": task_loss_map[args.task_name],
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
                    "| Epoch: {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.6e} | scale: {:10.0f} | time: {:.1f} |"
                    + " token/max: {:.3f} | mask/max: {:.3f} | grad_norm: {:.3f}"
                ).format(
                    epoch,
                    iteration,
                    args.train_iters,
                    task_loss_map[args.task_name],
                    lr_scheduler.current_lr,
                    int(optim_manager.loss_scale),
                    avg_time,
                    input_length.float().mean() / args.max_length,
                    (targets >= 0).sum(-1).float().mean() / args.max_length,
                    grad_norm,
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
                
                # save_folder_name = f"{args.save}{epoch}{iteration}"
                # model_fname = os.path.join(save_folder_name, f"{args.save_name}-iter-{iteration}.pt")
                # os.makedirs(os.path.dirname(model_fname), exist_ok=True)
                # if bmt.rank() == 0:
                #     shutil.copy(args.model_config, os.path.join(save_folder_name, "config.json"))
                #     shutil.copy(args.vocab, os.path.join(save_folder_name, "vocabs.txt"))

            if args.tensorboard is not None and bmt.rank() == 0:
                writer.add_scalar(f"Loss/train/{epoch}", task_loss_map[args.task_name], iteration)
                writer.add_scalar(f"Optimizer/lr/{epoch}", lr_scheduler.current_lr, iteration)
                writer.add_scalar(f"Optimizer/scale/{epoch}", optim_manager.loss_scale, iteration)
                writer.add_scalar(f"Optimizer/grad_norm/{epoch}", grad_norm.item(), iteration)
            
            # if iteration % 10 == 0:
            #     save_folder_name = f"{args.save}{epoch}"
            #     model_fname = os.path.join(save_folder_name, f"{args.save_name}-epoch-{epoch}.pt")
            #     os.makedirs(os.path.dirname(model_fname), exist_ok=True)
            #     bmt.save(model, model_fname)
            #     if bmt.rank() == 0:
            #         shutil.copy(args.model_config, os.path.join(save_folder_name, "config.json"))
            #         shutil.copy(args.vocab, os.path.join(save_folder_name, "vocabs.txt"))

        save_folder_name = f"{args.save}/epoch-{epoch}"
        model_fname = os.path.join(save_folder_name, f"{args.save_name}.pt")
        os.makedirs(os.path.dirname(model_fname), exist_ok=True)
        state_dict = model.state_dict()
        if args.delta_type == None or args.save_origin_model == True :
            print("saving base model...")
            bmt.save(model, model_fname)
        if args.delta_type != None and bmt.rank() == 0:
            print("saving delta model...")
            torch.save(state_dict, os.path.join(save_folder_name, f"{args.save_name}-delta.pt"))
        if bmt.rank() == 0:
            shutil.copy(args.model_config, os.path.join(save_folder_name, "config.json"))
            shutil.copy(args.vocab, os.path.join(save_folder_name, "vocabs.txt"))


def main():
    args = initialize()
    # To Be Specified
    bin_file = args.dataset
    bmt.print_rank(f"dataset: {bin_file}")
    
    tokenizer = get_tokenizer(args)
    dataloader = FinetuneDataset(
        bin_file, args.batch_size, args.max_length, tokenizer, unpad=(args.flash == "cuda"), task_name="task", drop_last=True
    )
    bmt.print_rank(f"#batch: {len(dataloader)}")
    total_steps = len(dataloader) * args.epoch
    
    setattr(args, 'train_iters', int(total_steps))
    setattr(args, 'warmup_iters', max(int(total_steps*0.02), args.warmup_iters))
    bmt.print_rank(json.dumps(vars(args), indent=2, sort_keys=True))
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    
    bmt.print_rank("finish loading")
    finetune(args, bin_file, tokenizer, model, optimizer, lr_scheduler)


if __name__ == "__main__":
    main()
