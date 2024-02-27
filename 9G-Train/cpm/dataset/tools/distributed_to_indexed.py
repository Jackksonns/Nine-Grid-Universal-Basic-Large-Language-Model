import argparse

import torch
from tqdm import tqdm

from cpm.dataset import SimpleDataset
from cpm.dataset.indexed_dataset import IndexedDatasetBuilder


def convert_cpm_data(cpm_path, out_path):
    dataset = SimpleDataset(cpm_path, shuffle=False)
    with IndexedDatasetBuilder(out_path, overwrite=True) as builder:
        for _ in tqdm(range(dataset._nlines), total=dataset._nlines):
            builder.put(dataset.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Data path in CPM format.")
    parser.add_argument("--output", "-o", required=True, help="Output data path in indexed jsonline format.")
    args = parser.parse_args()
    convert_cpm_data(args.input, args.output)
