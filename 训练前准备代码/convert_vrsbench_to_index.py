import argparse
import json
import logging
import os
import shutil
import h5py
from tqdm import tqdm

def calcu_data_length(data, language):
    """
    计算数据长度（token数量估算）
    """
    if language == "zh":
        return 0.7 * len(data)
    if language in ["en", "code"]:
        return 1.5 * len(data.split(" "))

def update_data(length, return_res):
    """
    更新数据长度分布统计
    """
    if length < 4000:
        return_res["less_4k"] += 1
    if length >= 4000 and length < 8000:
        return_res["4k-8k"] += 1
    if length >= 8000 and length < 16000:
        return_res["8k-16k"] += 1
    if length >= 16000 and length < 32000:
        return_res["16k-32k"] += 1
    if length >= 32000 and length < 64000:
        return_res["32k-64k"] += 1
    if length >= 64000 and length < 128000:
        return_res["64k-128k"] += 1
    if length >= 128000 and length < 256000:
        return_res["128k-256k"] += 1
    if length >= 256000:
        return_res["more_256k"] += 1
    return return_res

def check_string_length(json_obj, list_limit_length=1, str_limit_length=50):
    """
    限制sample样本的大小，用于meta.json中的data_sample字段
    """
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if isinstance(value, list) and len(value) > list_limit_length:
                json_obj[key] = value[:list_limit_length]
            elif isinstance(value, str):
                try:
                    value_json = json.loads(value)
                    check_string_length(value_json, list_limit_length, str_limit_length)
                    json_obj[key] = json.dumps(value_json, ensure_ascii=False)
                except json.JSONDecodeError:
                    if len(value) > str_limit_length:
                        json_obj[key] = value[:str_limit_length] + "..."
            elif isinstance(value, dict):
                check_string_length(value, list_limit_length, str_limit_length)
    elif isinstance(json_obj, list):
        for i in range(len(json_obj)):
            if isinstance(json_obj[i], list) and len(json_obj[i]) > list_limit_length:
                json_obj[i] = json_obj[i][:list_limit_length]
            elif isinstance(json_obj[i], str):
                try:
                    value_json = json.loads(json_obj[i])
                    check_string_length(value_json, list_limit_length, str_limit_length)
                    json_obj[i] = json.dumps(value_json, ensure_ascii=False)
                except json.JSONDecodeError:
                    if len(json_obj[i]) > str_limit_length:
                        json_obj[i] = json_obj[i][:str_limit_length] + "..."
            elif isinstance(json_obj[i], dict):
                check_string_length(json_obj[i], list_limit_length, str_limit_length)

def check_index_line(data_path: str, index_path: str, h5_path):
    """
    检查index文件和数据文件的行数是否一致
    """
    index_len = 0
    data_len = 0
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            data_len = sum(1 for line in file)
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as file:
            index_len = sum(1 for line in file)

    if index_len == data_len and index_len != 0:
        return index_len, data_len, True
    else:
        return index_len, data_len, False

def list_jsonl_files(directory):
    """
    列出目录下所有的jsonl文件
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

def build_index(file_list, output_path, language):
    """
    构建index文件
    """
    offset = 0
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    meta_path = os.path.join(output_path, "meta.json")
    data_path = os.path.join(output_path, "data.jsonl")
    nlines = 0
    data_sample = None
    all_token = 0
    starts = []
    length_distribute = {
        "less_4k": 0,
        "4k-8k": 0,
        "8k-16k": 0,
        "16k-32k": 0,
        "32k-64k": 0,
        "64k-128k": 0,
        "128k-256k": 0,
        "more_256k": 0,
    }
    
    w_index_h5 = h5py.File(os.path.join(output_path, "index.h5"), "w")
    w_index = open(os.path.join(output_path, "index"), "w")

    print(f"Processing data to: {data_path}")
    
    if os.path.exists(data_path):
        print("data.jsonl already exists! No need create data.jsonl!")
        with open(data_path, "rb") as fin:
            for idx, line in tqdm(enumerate(fin), desc="Processing existing data.jsonl"):
                nlines += 1
                offset += len(line)
                starts.append(offset)
                w_index.write(f"{offset}\n")
                decoded_line = line.decode("utf-8")
                if idx == 0:
                    data_sample = json.loads(decoded_line)
                    check_string_length(data_sample)
                length = calcu_data_length(decoded_line, language)
                all_token += length
                update_dic = update_data(length, length_distribute)
                length_distribute.update(update_dic)
    else:
        print("Creating data.jsonl...")
        w_data_path = open(data_path, "wb")
        
        for ds in tqdm(file_list, desc="Processing input files"):
            with open(ds, "rb") as fin:
                for idx, line in tqdm(enumerate(fin), desc=f"Processing {os.path.basename(ds)}"):
                    nlines += 1
                    offset += len(line)
                    starts.append(offset)
                    w_index.write(f"{offset}\n")
                    decoded_line = line.decode("utf-8")
                    if idx == 0 and data_sample is None:
                        data_sample = json.loads(decoded_line)
                        check_string_length(data_sample)
                    length = calcu_data_length(decoded_line, language)
                    all_token += length
                    update_dic = update_data(length, length_distribute)
                    length_distribute.update(update_dic)
                    w_data_path.write(line)
        w_data_path.close()

    if nlines == 0:
        print("Warning: No data processed!")
        return 0, 0, False

    # 创建meta.json文件
    meta_dic = {
        "language": language,
        "nlines": nlines,
        "nbytes": offset,
        "length_distribute": length_distribute,
        "avg_token_per_line": all_token / nlines if nlines > 0 else 0,
        "data_sample": data_sample,
    }
    
    with open(meta_path, "w", encoding='utf-8') as w:
        w.write(json.dumps(meta_dic, ensure_ascii=False, indent=2) + "\n")

    # 创建h5文件
    w_index_h5.create_dataset("index", data=starts)
    w_index_h5.close()
    w_index.close()
    
    return nlines, all_token, True

def process_vrsbench_data():
    """
    处理VRSBench数据的主函数
    """
    # 配置参数
    input_path = "./converted_data/vrsbench"
    output_base_path = "./indexed_data"
    
    # 检查输入目录是否存在
    if not os.path.exists(input_path):
        print(f"Error: Input directory {input_path} does not exist!")
        print("Please run prepare_vrsbench_data.py first to generate the jsonl files.")
        return
    
    # 获取所有jsonl文件
    jsonl_files = list_jsonl_files(input_path)
    if not jsonl_files:
        print(f"No jsonl files found in {input_path}")
        return
    
    print(f"Found {len(jsonl_files)} jsonl files:")
    for file in jsonl_files:
        print(f"  - {file}")
    
    # 为每个jsonl文件创建对应的index目录
    for jsonl_file in jsonl_files:
        # 获取相对路径，用于创建输出目录名
        rel_path = os.path.relpath(jsonl_file, input_path)
        file_name = os.path.splitext(rel_path)[0]  # 去掉.jsonl扩展名
        
        # 创建输出目录
        output_path = os.path.join(output_base_path, file_name)
        
        print(f"\nProcessing: {jsonl_file}")
        print(f"Output to: {output_path}")
        
        # 如果输出目录已存在，删除它
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        
        # 构建index
        nlines, all_token, success = build_index([jsonl_file], output_path, "en")
        
        if success:
            # 验证生成的文件
            output_jsonl = os.path.join(output_path, "data.jsonl")
            output_index = os.path.join(output_path, "index")
            output_h5 = os.path.join(output_path, "index.h5")
            output_meta = os.path.join(output_path, "meta.json")
            
            index_len, data_len, check_result = check_index_line(output_jsonl, output_index, output_h5)
            
            if check_result:
                print(f"✅ Successfully processed {file_name}")
                print(f"  - Lines: {nlines}")
                print(f"  - Total tokens: {all_token:.0f}")
                print(f"  - Average tokens per line: {all_token/nlines:.2f}")
                print(f"  - Generated files:")
                print(f"    * {output_jsonl}")
                print(f"    * {output_index}")
                print(f"    * {output_h5}")
                print(f"    * {output_meta}")
            else:
                print(f"❌ Error: Index and data file line counts don't match!")
                print(f"  - Index lines: {index_len}")
                print(f"  - Data lines: {data_len}")
        else:
            print(f"❌ Failed to process {file_name}")

def main():
    """
    主函数
    """
    print("VRSBench数据转换为Index格式")
    print("=" * 50)
    
    # 检查是否已经运行了prepare_vrsbench_data.py
    vrsbench_dir = "./converted_data/vrsbench"
    if not os.path.exists(vrsbench_dir):
        print("Error: VRSBench converted data not found!")
        print("Please run 'python prepare_vrsbench_data.py' first to generate the jsonl files.")
        return
    
    # 处理数据
    process_vrsbench_data()
    
    print("\n" + "=" * 50)
    print("转换完成！")
    print("所有index文件已保存到 ./indexed_data/ 目录下")

if __name__ == "__main__":
    main() 