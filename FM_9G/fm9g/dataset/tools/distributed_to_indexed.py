#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2023 AI, ZHIHU Inc. (zhihu.com)
#
# @author: ouzebin <ouzebin@zhihu.com>
# @date: 2023/07/27


import argparse

import torch
from tqdm import tqdm

from fm9g.dataset import SimpleDataset
from fm9g.dataset.indexed_dataset import IndexedDatasetBuilder


def convert_fm9g_data(fm9g_path, out_path):
    dataset = SimpleDataset(fm9g_path, shuffle=False)
    with IndexedDatasetBuilder(out_path, overwrite=True) as builder:
        for _ in tqdm(range(dataset._nlines), total=dataset._nlines):
            builder.put(dataset.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Data path in fm9g format.")
    parser.add_argument("--output", "-o", required=True, help="Output data path in indexed jsonline format.")
    args = parser.parse_args()
    convert_fm9g_data(args.input, args.output)
