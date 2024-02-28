import os

from libcpm import CPM9G

import argparse, json, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, help="the path of ckpt")
    parser.add_argument("--config", type=str, help="the path of config file")
    parser.add_argument("--vocab", type=str, help="the path of vocab file")
    args = parser.parse_args()

    model_config = json.load(open(args.config, 'r'))
    model_config["new_vocab"] = True

    model = CPM9G(
        "",
        args.vocab,
        0,
        memory_limit = 30 << 30,
        model_config=model_config,
        load_model=False,
    )
    model.load_model_pt(args.pt)

    datas = [
        '''<用户>马化腾是谁？<AI>''',
        '''<用户>你是谁？<AI>''',
        '''<用户>我要参加一个高性能会议，请帮我写一个致辞。<AI>''',
    ]

    # print(model.inference(datas, max_length=30))  # inference batch

    for data in datas:
        res = model.inference(data, max_length=4096)
        print(res['result'])
        # print(model.random_search(data))

if __name__ == "__main__":
    main()