import torch
import struct
import numpy as np


def write_string(fp, v):
    v = v.encode("utf-8")
    fp.write( struct.pack("I", len(v)) )
    fp.write(v)

def write_tuple(fp, v):
    fp.write( struct.pack("B", len(v)) )
    for i in v:
        fp.write( struct.pack("I", i) )

def write_dtype(fp, v):
    sv = -1
    if v == np.int8:
        sv = 0
    elif v == np.float16:
        sv = 1
    if sv == -1:
        raise TypeError("Unknown dtype %s" % v)
    fp.write( struct.pack("B", sv) )

def write_parameter(fp, name : str, value : torch.Tensor):
    write_string(fp, name)
    write_tuple(fp, value.size())
    value = np.ascontiguousarray(value.cpu().numpy())
    value_bytes = value.tobytes()
    fp.write( struct.pack("I", len(value_bytes)) )
    write_dtype(fp, value.dtype)
    fp.write(value_bytes)

def split(x, s):
    sizes = []
    for it in x.size():
        sizes.append(it)
    assert sizes[0] % s == 0
    sizes = [s, sizes[0] // s ] + sizes[1:]
    return x.reshape(*sizes)


def main(src_model_path, dst_model_path, layer_num):
  
    model = torch.load(src_model_path, map_location="cpu")
    params = {}

    params["input_embedding.weight"] = model["input_embedding.weight"].cpu()
    params["lm_head.weight"] = model["lm_head.weight"].cpu()
    params["output_layernorm.weight"] = (model["encoder.output_layernorm.weight"]).cpu()
    for i in range(layer_num):
        params[f"layers.{i}.ln_attn.weight"] = model[f"encoder.layers.{i}.self_att.layernorm_before_attention.weight"].cpu()

        params[f"layers.{i}.attn.project_q.weight"] = model[f"encoder.layers.{i}.self_att.self_attention.project_q.weight"]
        params[f"layers.{i}.attn.project_k.weight"] = model[f"encoder.layers.{i}.self_att.self_attention.project_k.weight"]
        params[f"layers.{i}.attn.project_v.weight"] = model[f"encoder.layers.{i}.self_att.self_attention.project_v.weight"]

        params[f"layers.{i}.attn.attn_out.weight"] = model[f"encoder.layers.{i}.self_att.self_attention.attention_out.weight"]

        params[f"layers.{i}.ln_ff.weight"] = model[f"encoder.layers.{i}.ffn.layernorm_before_ffn.weight"].cpu()

        params[f"layers.{i}.ff.w_in.weight"] = model[f"encoder.layers.{i}.ffn.ffn.w_in.w_0.weight"]
        params[f"layers.{i}.ff.w_gated.weight"] = model[f"encoder.layers.{i}.ffn.ffn.w_in.w_1.weight"]
        params[f"layers.{i}.ff.w_out.weight"] = model[f"encoder.layers.{i}.ffn.ffn.w_out.weight"]

    #转换后的模型
    fout = open(dst_model_path, "wb")
    fout.write( struct.pack("I", len(params)) )
    for name, value in params.items():
        write_parameter(fout, name, value)
    fout.close()



if __name__ == '__main__':
    # 输入已有的源模型
    src_model_path = "./checkpoints-epoch-1/cpm9g-8b-sft-epoch-1.pt"
    # 格式转换后的模型地址
    dst_model_path = "model_8b.ckpt"
    
    # 百亿：32
    # 千亿：80 
    layer_num = 32

    main(src_model_path, dst_model_path, layer_num)