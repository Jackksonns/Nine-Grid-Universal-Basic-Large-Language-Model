
import random


def rand(n: int, r: random.Random):
    return int(r.random() * n)

def transform(data, num_sample: int, r: random.Random):
    if 'input' in data:
        _input = data['input']
    else:
        _input = ""
    
    if 'output' in data:
        _output = data['output']
    else:
        _output = ""
    return {"input": _input, 
            "output": _output,
            }
