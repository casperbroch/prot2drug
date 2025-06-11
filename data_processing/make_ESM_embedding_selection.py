import time

import pandas as pd
import torch

PATH_SHORT = r"data\_esm_large\embeddings_7176_1481.pth"
PATH_LONG = r"data\_esm_large\embeddings_7478_3033.pth"

USEFULL_PATH = r"data\_esm_large\usefull_embeddings.pth"
LENGTH_PATH = r"data\_esm_large\usefull_embeddings_"


def run():
    select_usefull_proteins()
    select_protein_length(2000)

def select_usefull_proteins():
    """Select the protein embeddings that are actually in Papyrus++"""
    prot_ids = pd.read_csv(r"../data/papyrus/05.6++_combined_set_without_stereochemistry.tsv.xz", sep="\t")['target_id']

    short_dct = load_dict(PATH_SHORT)
    long_dct = load_dict(PATH_LONG)

    filtered_dict = {
        k: (short_dct[k] if k in short_dct else long_dct[k])
        for k in prot_ids if k in short_dct or k in long_dct
    }

    torch.save(filtered_dict, USEFULL_PATH)


def select_protein_length(threshold):
    """Makes a separate file with only the protein embeddings that are smaller than length threshold"""
    embeddings = load_dict(USEFULL_PATH)
    print(len(embeddings))
    thr_embeddings = {k: v for k, v in embeddings.items() if v.shape[0] <= threshold}
    print(len(thr_embeddings))
    torch.save(thr_embeddings, f"{LENGTH_PATH}{threshold}.pth")



def load_dict(path):
    """Loads the embeddings dictionary, where the Papyrus target_id values are the keys"""
    start_time = time.time()
    dct = torch.load(path)
    end_time = time.time()
    print(f"Load time is {end_time - start_time:.2f} seconds")
    return dct


if __name__ == "__main__":
    run()
