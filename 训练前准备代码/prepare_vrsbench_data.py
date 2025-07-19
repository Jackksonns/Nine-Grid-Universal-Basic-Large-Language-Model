import os
import json
import tqdm
import random
from datasets import load_dataset

# 设置随机种子确保结果可复现
random.seed(3014)

# 数据路径配置
vrsbench_dir = "./VRSBench"
output_path = "./converted_data/vrsbench"

# 创建输出目录
os.makedirs(output_path, exist_ok=True)

def is_english(text):
    """
    检查文本是否主要为英文
    """
    english_count = 0
    for char in text:
        if char.isalpha() and ord(char) < 128:
            english_count += 1
    return english_count / len(text) > 0.5 if len(text) > 0 else False

def process_vqa_data():
    """
    处理VQA（视觉问答）数据
    """
    print("Processing VQA data...")
    vqa_file = os.path.join(vrsbench_dir, "VRSBench_EVAL_vqa.json")
    
    if not os.path.exists(vqa_file):
        print(f"VQA file not found: {vqa_file}")
        return
    
    # 读取VQA数据
    with open(vqa_file, 'r', encoding='utf-8') as f:
        vqa_data = json.load(f)
    
    # 转换为训练格式
    vqa_train_data = []
    for item in tqdm.tqdm(vqa_data, desc="Processing VQA"):
        if is_english(item['question']) and is_english(item['ground_truth']):
            # 构建对话格式
            conversation = [
                {
                    'role': 'User',
                    'content': f"<image>\n[question] {item['question']}"
                },
                {
                    'role': 'Assistant', 
                    'content': item['ground_truth']
                }
            ]
            vqa_train_data.append({
                'image': item['image_id'],
                'conversations': conversation
            })
    
    # 保存处理后的数据
    vqa_output_file = os.path.join(output_path, 'vqa_train.jsonl')
    with open(vqa_output_file, 'w', encoding='utf-8') as f:
        for item in vqa_train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"VQA data processed: {len(vqa_train_data)} samples saved to {vqa_output_file}")

def process_referring_data():
    """
    处理Referring（目标引用）数据
    """
    print("Processing Referring data...")
    referring_file = os.path.join(vrsbench_dir, "VRSBench_EVAL_referring.json")
    
    if not os.path.exists(referring_file):
        print(f"Referring file not found: {referring_file}")
        return
    
    # 读取Referring数据
    with open(referring_file, 'r', encoding='utf-8') as f:
        referring_data = json.load(f)
    
    # 转换为训练格式
    referring_train_data = []
    for item in tqdm.tqdm(referring_data, desc="Processing Referring"):
        if is_english(item['question']) and is_english(item['ground_truth']):
            # 构建对话格式
            conversation = [
                {
                    'role': 'User',
                    'content': f"<image>\n[referring] {item['question']}"
                },
                {
                    'role': 'Assistant',
                    'content': item['ground_truth']
                }
            ]
            referring_train_data.append({
                'image': item['image_id'],
                'conversations': conversation
            })
    
    # 保存处理后的数据
    referring_output_file = os.path.join(output_path, 'referring_train.jsonl')
    with open(referring_output_file, 'w', encoding='utf-8') as f:
        for item in referring_train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Referring data processed: {len(referring_train_data)} samples saved to {referring_output_file}")

def process_caption_data():
    """
    处理Caption（图像描述）数据
    """
    print("Processing Caption data...")
    caption_file = os.path.join(vrsbench_dir, "VRSBench_EVAL_Cap.json")
    
    if not os.path.exists(caption_file):
        print(f"Caption file not found: {caption_file}")
        return
    
    # 读取Caption数据
    with open(caption_file, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)
    
    # 转换为训练格式
    caption_train_data = []
    for item in tqdm.tqdm(caption_data, desc="Processing Caption"):
        if is_english(item['ground_truth']):
            # 构建对话格式
            conversation = [
                {
                    'role': 'User',
                    'content': "<image>\n[caption] Could you describe the contents of this image for me?"
                },
                {
                    'role': 'Assistant',
                    'content': item['ground_truth']
                }
            ]
            caption_train_data.append({
                'image': item['image_id'],
                'conversations': conversation
            })
    
    # 保存处理后的数据
    caption_output_file = os.path.join(output_path, 'caption_train.jsonl')
    with open(caption_output_file, 'w', encoding='utf-8') as f:
        for item in caption_train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Caption data processed: {len(caption_train_data)} samples saved to {caption_output_file}")

def process_training_data():
    """
    处理训练数据（包含所有任务的综合数据）
    """
    print("Processing training data...")
    train_file = os.path.join(vrsbench_dir, "VRSBench_train.json")
    
    if not os.path.exists(train_file):
        print(f"Training file not found: {train_file}")
        return
    
    # 读取训练数据
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 转换为训练格式
    train_processed_data = []
    for item in tqdm.tqdm(train_data, desc="Processing Training Data"):
        # 检查对话是否包含英文内容
        has_english = False
        for conv in item['conversations']:
            if is_english(conv['value']):
                has_english = True
                break
        
        if has_english:
            # 转换对话格式
            conversations = []
            for conv in item['conversations']:
                if conv['from'] == 'human':
                    conversations.append({
                        'role': 'User',
                        'content': conv['value']
                    })
                elif conv['from'] == 'gpt':
                    conversations.append({
                        'role': 'Assistant',
                        'content': conv['value']
                    })
            
            train_processed_data.append({
                'image': item['image'],
                'conversations': conversations
            })
    
    # 保存处理后的数据
    train_output_file = os.path.join(output_path, 'train.jsonl')
    with open(train_output_file, 'w', encoding='utf-8') as f:
        for item in train_processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Training data processed: {len(train_processed_data)} samples saved to {train_output_file}")

def create_combined_dataset():
    """
    创建综合数据集，包含所有任务类型
    """
    print("Creating combined dataset...")
    
    combined_data = []
    
    # 读取所有处理后的数据文件
    data_files = [
        os.path.join(output_path, 'vqa_train.jsonl'),
        os.path.join(output_path, 'referring_train.jsonl'),
        os.path.join(output_path, 'caption_train.jsonl'),
        os.path.join(output_path, 'train.jsonl')
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        combined_data.append(json.loads(line))
    
    # 随机打乱数据
    random.shuffle(combined_data)
    
    # 划分训练集和验证集
    split_idx = int(len(combined_data) * 0.95)
    train_data = combined_data[:split_idx]
    val_data = combined_data[split_idx:]
    
    # 保存综合数据集
    combined_train_file = os.path.join(output_path, 'combined_train.jsonl')
    combined_val_file = os.path.join(output_path, 'combined_val.jsonl')
    
    with open(combined_train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(combined_val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Combined dataset created:")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print(f"  - Total samples: {len(combined_data)}")

def main():
    """
    主函数：执行所有数据处理步骤
    """
    print("开始处理VRSBench数据集...")
    print(f"数据源目录: {vrsbench_dir}")
    print(f"输出目录: {output_path}")
    
    # 处理各种任务类型的数据
    process_vqa_data()
    process_referring_data()
    process_caption_data()
    process_training_data()
    
    # 创建综合数据集
    create_combined_dataset()
    
    print("\n数据处理完成！")
    print(f"所有处理后的数据已保存到: {output_path}")
    print("\n生成的文件包括:")
    print("  - vqa_train.jsonl: VQA任务数据")
    print("  - referring_train.jsonl: Referring任务数据") 
    print("  - caption_train.jsonl: Caption任务数据")
    print("  - train.jsonl: 原始训练数据")
    print("  - combined_train.jsonl: 综合训练数据集")
    print("  - combined_val.jsonl: 综合验证数据集")

if __name__ == "__main__":
    main() 