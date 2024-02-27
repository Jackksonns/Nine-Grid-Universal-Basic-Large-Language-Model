import bmtrain as bmt

from ...dataset import SimpleDataset
from ..tokenizers import CPM9GTokenizer
from .pretrain import _MixedDatasetBatchPacker
from .pretrain import _MixedDatasetConfig
from .pretrain import CPM9GBatch


class FinetuneDataset:
    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        max_length: int,
        tokenizer: CPM9GTokenizer,
        unpad: bool = False,
        task_name: str = "task",
        drop_last: bool = False,
    ) -> None:
        self._world_size = bmt.world_size()
        self._rank = bmt.rank()
        self._batch_size = batch_size
        self._unpad = unpad
        self._max_length = max_length

        self._packer = _MixedDatasetBatchPacker(batch_size * self._world_size, max_length, tokenizer, unpad)
        self._drop_last = drop_last

        ds = SimpleDataset(dataset_path, shuffle=False)
        self._ds_cfg: _MixedDatasetConfig = {
            "weight": 1.0,
            "path": dataset_path,
            "transforms": [],
            "task_name": task_name,
            "dataset_name": "finetune",
            "incontext_weight": [1.0],
            "lines": len(ds),
            "dataset": ds,
        }
        self._nlines = len(ds)
        self._nbytes = ds.get_bytes()

    def __batch_iter(self):
        while True:
            try:
                batch = self._packer.add_data(self._ds_cfg)
            except EOFError:
                break
            if batch is None:
                continue
            yield batch
        if len(self._packer) > 0:
            batch = self._packer.pack_batch(force=True)
            if not self._drop_last:
                yield batch
        self._ds_cfg["dataset"]._repeat_times = 0

    def __iter__(self):
        if not self._unpad:
            batch_st = self._batch_size * self._rank
            batch_end = self._batch_size * (self._rank + 1)
            for batch in self.__batch_iter():
                batch_size = batch["inputs"].shape[0]
                if batch_size <= batch_st:
                    yield None
                else:
                    ret: CPM9GBatch = {
                        kw: val[batch_st:batch_end]  # type: ignore
                        for kw, val in batch.items()
                        if kw not in ["task_names", "raw_data", "cu_seqlens", "max_seqlen"]
                    }  # type: ignore
                    ret["task_names"] = batch["task_names"]
                    ret["raw_data"] = batch["raw_data"]
                    ret["cu_seqlens"] = batch["cu_seqlens"]
                    ret["max_seqlen"] = batch["max_seqlen"]
                    yield ret
        else:
            for batch in self.__batch_iter():
                assert batch["inputs"].shape[0] == 1
                yield batch

    def __len__(self):
        iter_count = int(self._nbytes // (3.6 * self._batch_size * self._world_size * self._max_length))
        if not self._drop_last:
            iter_count += 1
        return iter_count