import os
import struct
import json
from typing import List

import libcpm
from flask import Flask, Response, request

# from concurrent.futures import ThreadPoolExecutor
# executor = ThreadPoolExecutor(1)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def _load_dtype(fp):
    dtype = struct.unpack("B", fp.read(1))[0]
    return dtype

def _load_string(fp):
    size = struct.unpack("I", fp.read(4))[0]
    return fp.read(size).decode("utf-8")

def _load_tuple(fp):
    ndim = struct.unpack("B", fp.read(1))[0]
    ret = []
    for i in range(ndim):
        ret.append(struct.unpack("I", fp.read(4))[0])
    return tuple(ret)

class LocalLoader(libcpm.ModelLoader):
    def __init__(self,
            model_path : str,
            vocab_path : str,
        ):
        vocabs = []
        with open(vocab_path, "r") as fin:
            for line in fin:
                if line.startswith("\""):
                    vocabs.append(json.loads(line))
        self._vocabs = vocabs
        # print(len(vocabs), "tokens")

        with open(model_path, "rb") as fp:
            num_parameters = struct.unpack("I", fp.read(4))[0]
            parameters = {}
            for _ in range(num_parameters):
                param_name = "model." + _load_string(fp)
                _ = _load_tuple(fp)
                param_size = struct.unpack("I", fp.read(4))[0]
                _ = _load_dtype(fp)
                param = fp.read(param_size)
                parameters[param_name] = param
        self._parameters = parameters

    def fetch_parameter(self, name):
        # print(name, len(self._parameters[name]))
        return self._parameters[name]

    @property
    def num_layers(self):
        return 32

    @property
    def dim_model(self):
        return 4096

    @property
    def num_heads(self):
        return 32

    @property
    def num_kv_heads(self):
        return 32

    @property
    def dim_head(self):
        return 128

    @property
    def dim_ff(self):
        return 14336

    @property
    def tokens(self):
        return self._vocabs

    @property
    def rope_theta(self):
        return 10000.0



model = libcpm.CPMCaterpillar(
    #add converted model and vocabs
    LocalLoader(
        "model_8b.ckpt",
        "vocabs.txt",
    ),
    memory_limit = 40 << 30,
)

app = Flask(__name__)
import logging
logging.basicConfig(filename='error_8b.log',level=logging.DEBUG)

@app.route("/llm", methods=["get", "post"])
def llm():
    content: str = request.json["content"]
    if "params" in request.json:
        params = request.json["params"]
    else:
        params = {}
    # ret = executor.submit(_llm, content).result()
    ret = _llm(content, params)
    return ret

def _llm(content, params):
    logging.debug("~ content:\n" + content)
    logging.debug("~ input_params:\n" + json.dumps(params, ensure_ascii=False))

    def generate_events(content):
        ipt = content.replace("<用户>", "<sep>用户：")
        ipt = ipt.replace("<AI>", "<sep>AI：")
        ipt = ipt.lstrip("<sep>")
        old_ans = ""
        logging.debug("~ ans:")
        true_params = {}
        USING_PARAMS = {"max_length", "repetition_penalty", "ngram_penalty", "seed", "temperature", "top_p", "top_k", "interval"}
        true_params = {}
        for p in USING_PARAMS:
            if p in params:
                true_params[p] = params[p]
        if "max_length" not in true_params:
            true_params["max_length"] = 4096

        logging.debug("~ true_params:\n" + json.dumps(true_params, ensure_ascii=False))
        for it in model.random_search(ipt, **true_params):
            ans = it["result"]
            if ans is not None:
                return_data = "data:" + json.dumps({"text": ans[len(old_ans):]}, ensure_ascii=False) + "\n\n"
                yield return_data
                logging.debug("return_data[" + return_data.strip() + "]")
                old_ans = ans
            if it["stoped"]:
                break
        logging.debug("\n")
    return Response(generate_events(content), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=True, use_reloader=False)

