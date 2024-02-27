import random


def rand(n: int, r: random.Random):
    return int(r.random() * n)


def transform(data, num_sample: int, r: random.Random):
    return {"input": "", "output": data["text"]}
