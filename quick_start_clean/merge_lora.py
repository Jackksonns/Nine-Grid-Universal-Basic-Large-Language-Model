import torch
import os
import sys
import shutil
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="base model path", required=True)
    parser.add_argument("--delta_path", type=str, help="the lora model path", required=True)
    parser.add_argument("--merge_path", type=str, help="merge the base and lora model as one models", required=True)


    args = parser.parse_args()
    return args


def merge_lora_models(args):

    scale = 64

    # model = torch.load(f"/home/wangxvjia/9g_models/llama_fin_new/checkpoints-epoch-4/llama-finance-sft-iteration-258-delta.pt")
    model = torch.load(args.delta_path)

    dic = {}
    num = 0
    allocated_mem = torch.cuda.memory_allocated()
    print(f"allocated GPU memory: {allocated_mem/1024**3} GB")
    for key, value in model.items():
        print(key)
        print(value.shape)
        layer_list = key.split('.')
        layer = ".".join(layer_list[:-1])
        if layer in dic:
            other = dic[layer].cuda()
            value = value.cuda()
            if layer_list[-1] == "lora_B":
                other = torch.mm(value, other).cpu()
                alpha = scale / value.shape[1]
            else :
                other = torch.mm(other, value).cpu()
            dic.update({layer: other})
        else:
            dic.update({layer: value})
    print("end")
    print(f"alpha: {scale} | weight: {alpha}")

    torch.cuda.empty_cache()
    print("begin")
    base_model = torch.load(args.base_path ,map_location=torch.device('cpu'))
    # base_model = torch.load("/data/public/opensource_models/meta-llama/Llama-2-7b-mc/pytorch_model.pt",map_location=torch.device('cpu'))
    
    for key, value in base_model.items():
        layer_list = key.split('.')
        layer = ".".join(layer_list[:-1]) + ".lora"
        value = value.cuda()
        if layer in dic:
            print(layer)
            other = dic[layer].cuda()
            value = torch.add(value,  alpha * other.half()).detach().cpu()
            print(value)
        value = value.cpu()
        base_model.update({key: value})

    # torch.save(base_model, f"/home/wangxvjia/9g_models/cpm_fin_new_1e4/fin/pytorch_model.pt")
    torch.save(base_model, args.merge_path)

    exit(0)

if __name__=='__main__':

    args = get_args()
    merge_lora_models(args)