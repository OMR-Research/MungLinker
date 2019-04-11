from typing import List, Dict

from muscima.cropobject import CropObject
from tqdm import tqdm

from munglinker.data_pool import PairwiseMungoDataPool


def count_relationships_by_type(data: Dict[str, PairwiseMungoDataPool]):
    relationship_counter = {}
    dicts = ["train", "valid", "test"]
    for data_dict in dicts:
        from munglinker.batch_iterators import PoolIterator
        iterator = PoolIterator(1,shuffle=False)
        for data_batch in tqdm(iterator(data[data_dict]), total=(len(data[data_dict]))):
            mungos_from = data_batch["mungos_from"]  # type: List[CropObject]
            mungos_to = data_batch["mungos_to"]  # type: List[CropObject]
            name = mungos_from[0].clsname + " to " + mungos_to[0].clsname
            if name in relationship_counter:
                relationship_counter[name] = relationship_counter[name] + 1
            else:
                relationship_counter[name] = 1

    for relationship,count in relationship_counter.items():
        print("{0}: {1}".format(relationship, count))

