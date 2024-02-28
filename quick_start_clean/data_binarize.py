import concurrent
import os
import threading

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import time
import json
import random
import sys
import re
from tqdm import tqdm

sys.path.insert(0, '/data/public/CPM-9G/9G-Train')
from cpm.dataset import build_dataset, SimpleDataset
sys.setrecursionlimit(2000)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="raw dataset path", required=True)
    parser.add_argument("--data_type", type=str, help="datatype can be json or txt", required=True)
    parser.add_argument("--output_path", type=str, help="output dataset path", required=True)
    parser.add_argument("--output_name", type=str, help="output dataset name", required=True)
    parser.add_argument("--repair_keys", type=str, help="json string to json", required=False,
                        default="")

    args = parser.parse_args()
    return args


DROP_LEN = 100  ###### if the length of a piece of data is less than drop_len, drop it
MAX_LENGTH = 10000  ######the max length of final data
MAX_RECURSION = 1000


def split_sent(data_, depth):
    # 用于纯文本切分
    if len(data_) < DROP_LEN:
        return []
    if len(data_) > MAX_LENGTH and depth < MAX_RECURSION:

        if '\n' not in data_:
            return [{"text": data_}]
            # return [data_]
        mid = int(len(data_) / 2)
        while mid > 0 and (data_[mid - 1] not in ["\n", "。"]):  ######\n, 。等分割符号，根据自己需要改
            mid -= 1
        ret = []
        ret.extend(split_sent(data_[:mid], depth + 1))
        ret.extend(split_sent(data_[mid:], depth + 1))
        return ret
    else:
        # return [data_]
        return [{"text": data_}]


# import orjson
import time

import fcntl

TARGET_LANG = {"julia", "visual-basic", "java", "python", "c-sharp", "c", "cpp", "scala", "javascript", "go", "rust",
               "ruby", "haskell", "typescript", "lua", "php", "fortran", "r", "sql", "jupyter-scripts-julia",
               "jupyter-scripts-java", "jupyter-scripts-python", "jupyter-scripts-csharp", "jupyter-scripts-c++",
               "jupyter-scripts-scala", "jupyter-scripts-javascript", "jupyter-scripts-rust", "jupyter-scripts-haskell",
               "jupyter-scripts-typescript", "jupyter-scripts-R", "jupiter-structured", "git-commit"}


def is_target_lang(json_obj):
    return json_obj.get("clean_content", {}).get("lang", "") in TARGET_LANG


def deal_json_file(file_path, ds_write, repair_keys=None):
    print(f"begin deal {file_path}")
    t0 = time.time()
    with open(file_path, "r", encoding='utf-8') as fin:
        data_buffer = []
        for line in fin:
            line = line.strip()
            data = load_and_repair_json_string(line, repair_keys)
            data_buffer.append(data)
            if len(data_buffer) > 64:
                global T_LOCK
                if T_LOCK:
                    T_LOCK.acquire()
                    for data in data_buffer:
                        ds_write.write(data)
                    T_LOCK.release()
                else:
                    for data in data_buffer:
                        ds_write.write(data)
                data_buffer = []
        if T_LOCK:
            T_LOCK.acquire()
            for data in data_buffer:
                ds_write.write(data)
            T_LOCK.release()
    print(f"deal {os.path.basename(file_path)} time spend {time.time() - t0}")


def load_and_repair_json_string(line, repair_keys=None):
    data = json.loads(line)
    if repair_keys:
        for key in repair_keys:
            if data[key] is not None and isinstance(data[key], str):
                data[key] = json.loads(data[key])
    return data


T_LOCK = None


def main():
    args = get_args()
    file_list = []
    for file_i in os.listdir(args.input):
        tmp_dir = os.path.join(args.input, file_i)
        if os.path.isfile(tmp_dir):
            file_list.append(tmp_dir)
        else:
            for file_i_i in os.listdir(tmp_dir):
                file_list.append(os.path.join(tmp_dir, file_i_i))
    repair_keys = args.repair_keys.strip().split(",")
    if len(repair_keys) == 1 and repair_keys[0] == '':
        repair_keys = None

    file_list.sort()
    t0 = time.time()
    with build_dataset(args.output_path, args.output_name) as dataset:
        if args.data_type == "txt":
            for ds in file_list:
                print(ds)
                with open(ds, "r", encoding='utf-8') as fin:
                    for line in tqdm(fin):
                        line = json.loads(line.strip())
                        line = line.strip().replace("\\r\\n", "\n")
                        line = line.strip().replace("\\r", "\n")
                        line = line.strip().replace("<n>", "\n")
                        line = line.strip().replace("\\n", "\n")  ######清洗步骤根据自己需要改
                        line = re.sub('\n\n[\n]+', '\n\n', line.strip())
                        line = re.sub('(\n\s+\n)+', '\n\n', line.strip())
                        line_list = split_sent(line, 1)  #######递归切分line
                        for item in line_list:
                            dataset.write(item)
                        # dataset.write({"text":line})
        elif args.data_type == "json":
            global T_LOCK
            T_LOCK = threading.Lock()
            thread_pool = ThreadPoolExecutor(max_workers=1)
            tasks = []
            for ds_path in file_list:
                # deal_json_file(ds_path, dataset, repair_keys)
                tasks.append(thread_pool.submit(deal_json_file, *(ds_path, dataset, repair_keys)))
            wait(tasks)
            for task in tasks:
                if task.result():
                    pass
        print(f"all time spend:{time.time() - t0}")


if __name__ == "__main__":
    main()
