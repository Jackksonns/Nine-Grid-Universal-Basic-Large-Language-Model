import argparse
import json
import logging
import os
import shutil
import h5py
from tqdm import tqdm

#HDFS_INDEX_DATA_PATH = "/user/tc_agi/llm/index_datasets/"


def calcu_data_length(data, language):
    if language == "zh":
        return 0.7 * len(data)
    if language in ["en", "code"]:
        return 1.5 * len(data.split(" "))


def update_data(length, return_res):
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


# 限制sample样本的大小
def check_string_length(json_obj, list_limit_length=1, str_limit_length=50):
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if isinstance(value, list) and len(value) > list_limit_length:
                json_obj[key] = value[:list_limit_length]  # 截断列表长度
            elif isinstance(value, str):
                try:
                    value_json = json.loads(value)  # 尝试解析字符串为 JSON 对象
                    check_string_length(value_json, list_limit_length, str_limit_length)  # 递归处理 JSON 对象
                    json_obj[key] = json.dumps(value_json,ensure_ascii=False)  # 将修改后的 JSON 对象转回字符串
                except json.JSONDecodeError:
                    if len(value) > str_limit_length:
                        json_obj[key] = value[:str_limit_length] + "..."  # 截断字符串长度
            elif isinstance(value, dict):
                check_string_length(value, list_limit_length, str_limit_length)  # 递归遍历嵌套的字典或列表
    elif isinstance(json_obj, list):
        for i in range(len(json_obj)):
            if isinstance(json_obj[i], list) and len(json_obj[i]) > list_limit_length:
                json_obj[i] = json_obj[i][:list_limit_length]  # 截断列表长度
            elif isinstance(json_obj[i], str):
                try:
                    value_json = json.loads(json_obj[i])  # 尝试解析字符串为 JSON 对象
                    check_string_length(value_json, list_limit_length, str_limit_length)  # 递归处理 JSON 对象
                    json_obj[i] = json.dumps(value_json,ensure_ascii=False)  # 将修改后的 JSON 对象转回字符串
                except json.JSONDecodeError:
                    if len(json_obj[i]) > str_limit_length:
                        json_obj[i] = json_obj[i][:str_limit_length] + "..."  # 截断字符串长度
            elif isinstance(json_obj[i], dict):
                check_string_length(json_obj[i], list_limit_length, str_limit_length)  # 递归遍历嵌套的字典或列表


def check_index_line(data_path:str, index_path:str, h5_path):
    index_len = 0
    data_len = 0
    if os.path.exists(data_path):
        with open(data_path, 'r') as file:
            # 逐行迭代文件对象
            data_len = sum(1 for line in file)
    if os.path.exists(index_path):
        with open(data_path, 'r') as file:
            # 逐行迭代文件对象
            index_len = sum(1 for line in file)

    if index_len == data_len and index_len != 0:
        return index_len, data_len, True
    else:
        return index_len, data_len, False


def list_files_in_directory(directory):
    file_list = []

    # 遍历指定目录下的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.json', '.jsonl')) and "meta.json" not in file:
                file_path = os.path.join(root, file)
                file_list.append(file_path)

    return file_list

def build_index(file_list, output_path, language):
    offset = 0
    if not os.path.exists(output_path):
        os.mkdir(output_path)
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

    print(data_path)
    if os.path.exists(data_path):
        print("data.jsonl already exists! No need create data.jsonl!")
        with open(data_path, "rb") as fin:
            for idx, line in tqdm(enumerate(fin)):
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
        print("data.jsonl not exists! Creating data.jsonl...")
        w_data_path = open(data_path, "wb")
        
        #for ds in tqdm(file_list):
        if 1:
            with open(file_list, "rb") as fin:
                for idx, line in tqdm(enumerate(fin)):
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
                    w_data_path.write(line)
        w_data_path.close()

    if nlines == 0:
        return 0, 0, False

    assert os.path.exists(output_path), f"Jsonline dataset '{data_path}' not found."
    meta_dic = {
        "language": language,
        "nlines": nlines,
        "nbytes": offset,
        "length_distribute": length_distribute,
        "avg_token_per_line": all_token / nlines,
        "data_sample": data_sample,
    }
    with open(meta_path, "w") as w:
        w.write(json.dumps(meta_dic, ensure_ascii=False) + "\n")

    w_index_h5.create_dataset("index", data=starts)

    w_index.close()

if __name__ == "__main__":
    # python trans.py -p /home/jeeves/souyun/ -l zh -o /home/jeeves/souyunout -hdfs souyunout
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True, help="Input data path.")
    parser.add_argument("--language", "-l", required=True, help="language type, select 'zh' or 'en'")
    parser.add_argument("--output", "-o", required=True, help="Output index Data dir path.")
    #parser.add_argument("--hdfs_name", "-hdfs", required=True, help="hdfs save name. hdfs dir is '/user/tc_agi/llm/index_datasets/'")
    args = parser.parse_args()

    #file_list = list_files_in_directory(args.path)

    file_list = args.path
    #print(file_list)
    # if os.path.isfile(args.path):
    #     file_list = [args.path]
    # else:
    #     for file_namei in os.listdir(args.path):
    #         tmp_dir = os.path.join(args.path, file_namei)
    #         if os.path.isfile(tmp_dir):
    #             file_list.append(tmp_dir)
    #         else:
    #             for file_name_2 in os.listdir(tmp_dir):
    #                 file_list.append(os.path.join(tmp_dir, file_name_2))
    #file_list.sort()


    if os.path.exists(f"{args.output}"):
        shutil.rmtree(f"{args.output}")

    build_index(file_list, args.output, args.language)

    output_jsonl = f"{args.output}/data.jsonl"
    output_index = f"{args.output}/index"
    output_h5 = f"{args.output}/index.h5"
    output_meta = f"{args.output}/meta.json"
    index_len, data_len, check_result = check_index_line(output_jsonl, output_index, output_h5)
    '''
    if check_result:
        #上传hdfs
        os.system("hadoop fs -get -f hdfs://hdfs01:8020/etc/infra-conf-admin/packages/hdfs-copy-1.0-SNAPSHOT.jar .")
        os.system("export HADOOP_USER_NAME=tc_agi")
        os.system("export HADOOP_USER_PASSWORD=IH2U3AS1D")
        os.system(f"hadoop jar hdfs-copy-1.0-SNAPSHOT.jar com.zhihu.platform.Main --dst hdfs://hdfs01:8020{HDFS_INDEX_DATA_PATH}{args.hdfs_name}/data.jsonl --src file://{output_jsonl} --parallelism 50  --dstToken tc_agi-IH2U3AS1D")
        os.system(f"hadoop jar hdfs-copy-1.0-SNAPSHOT.jar com.zhihu.platform.Main --dst hdfs://hdfs01:8020{HDFS_INDEX_DATA_PATH}{args.hdfs_name}/index --src file://{output_index} --parallelism 50  --dstToken tc_agi-IH2U3AS1D")
        os.system(f"hadoop jar hdfs-copy-1.0-SNAPSHOT.jar com.zhihu.platform.Main --dst hdfs://hdfs01:8020{HDFS_INDEX_DATA_PATH}{args.hdfs_name}/index.h5 --src file://{output_h5} --parallelism 50  --dstToken tc_agi-IH2U3AS1D")
        os.system(f"hadoop jar hdfs-copy-1.0-SNAPSHOT.jar com.zhihu.platform.Main --dst hdfs://hdfs01:8020{HDFS_INDEX_DATA_PATH}{args.hdfs_name}/meta.json --src file://{output_meta} --parallelism 50  --dstToken tc_agi-IH2U3AS1D")
        print(f"index文件行数：{index_len}行，数据文件行数：{data_len}行")

    else:
        logging.error("==========data.jsonl和data_index行数不一致==========")
        logging.error(f"index文件行数：{index_len}行，数据文件行数：{data_len}行")
    '''
