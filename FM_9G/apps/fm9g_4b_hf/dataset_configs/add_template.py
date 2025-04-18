import random

def rand(n: int, r: random.Random):
    return int(r.random() * n)

def transform(data, num_sample: int, r: random.Random):
    if 'input' in data:
        user_input = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>User\n{}<|im_end|>\n<|im_start|>assistant\n".format(data['input'])
        ai_output = "{}<|im_end|>".format(data['output'])
    elif 'output' in data:
        user_input = ""
        ai_output = data['output']
    else:
        user_input = ''
        ai_output = ''

    return {
        "input": user_input, 
        "output": ai_output,
        }