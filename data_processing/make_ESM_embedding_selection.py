import sys
import time

import torch

from data_processing.papyrus_eda import get_data

PATH_SHORT = r"data\_esm_large\embeddings_7176_1481.pth"
PATH_LONG = r"data\_esm_large\embeddings_7478_3033.pth"

USEFULL_PATH = r"data\_esm_large\usefull_embeddings.pth"
LENGTH_PATH = r"data\_esm_large\usefull_embeddings_"


def run():
    # select_usefull_proteins()

    # select_protein_length(400)
    # select_protein_length(500)
    # select_protein_length(750)
    # select_protein_length(1000)
    select_protein_length(1500)

def select_usefull_proteins():
    prot_ids = get_data(data_type='mol')['target_id']

    short_dct = load_dict(PATH_SHORT)
    long_dct = load_dict(PATH_LONG)

    filtered_dict = {
        k: (short_dct[k] if k in short_dct else long_dct[k])
        for k in prot_ids if k in short_dct or k in long_dct
    }

    torch.save(filtered_dict, USEFULL_PATH)


def select_protein_length(threshold):
    """prot_ids = get_data(data_type='mol')['target_id']
    prot_length = get_data(data_type='prot')
    prot_length[prot_length['target_id'].isin(prot_ids)]"""
    # analyse_protein_distribution(prot_length)

    embeddings = load_dict(USEFULL_PATH)
    print(len(embeddings))
    thr_embeddings = {k: v for k, v in embeddings.items() if v.shape[0] <= threshold}
    print(len(thr_embeddings))
    torch.save(thr_embeddings, f"{LENGTH_PATH}{threshold}.pth")



def load_dict(path):
    start_time = time.time()
    dct = torch.load(path)
    end_time = time.time()
    print(f"Load time is {end_time - start_time:.2f} seconds")
    print(f"Memory is {total_size(dct)}")
    return dct


def total_size(o, seen=None):
    """Recursively finds size of objects, including contents."""
    if seen is None:
        seen = set()
    obj_id = id(o)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(o)
    if isinstance(o, dict):
        size += sum(total_size(k, seen) + total_size(v, seen) for k, v in o.items())
    elif isinstance(o, (list, tuple, set, frozenset)):
        size += sum(total_size(i, seen) for i in o)
    return size


if __name__ == "__main__":
    run()
