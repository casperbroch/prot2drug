import os
import sys

import numpy as np
import pandas as pd
import torch

from papyrus_eda import analyse_protein_distribution

PATHWAY_PATH = r"data\synthetic_pathways"
FILTERED_SMILES_PATH = os.path.join(PATHWAY_PATH, "filtered_pathways.pth")

PROTEIN_PATH = r"data\protein_embeddings"
FILTERED_PROTEIN_PATH1 = os.path.join(PROTEIN_PATH, "usefull_embeddings.pth")
FILTERED_PROTEIN_PATH2 = os.path.join(PROTEIN_PATH, "embeddings_selection_float16.pth")

PAPYRUS_PATH = r"data/papyrus"
PAPYRUS_SMILES = os.path.join(PAPYRUS_PATH, "05.6++_combined_set_without_stereochemistry.tsv.xz")
PAPYRUS_PROTEIN = os.path.join(PAPYRUS_PATH, "05.6_combined_set_protein_targets.tsv.xz")
PAPYRUS_FILTERED = os.path.join(PAPYRUS_PATH, "papyrus_selection.csv")

def run():
    # analyse_pathway_similarity()
    # filter_pathways()

    # filter_proteins()

    # filter_papyrus()

    papyrus = load_papyrus_processed()
    print(papyrus)
    print(papyrus.shape)


def analyse_pathway_similarity():
    np.set_printoptions(threshold=np.inf)

    pathway_files = os.listdir(PATHWAY_PATH)
    pathway_files = [pf for pf in pathway_files if pf[:10] == "embeddings"]
    similarities = []
    for pathway_file in pathway_files:
        print(f"Start with {pathway_file}")
        dct = torch.load(os.path.join(PATHWAY_PATH, pathway_file))
        for smiles, pathway_list in dct.items():
            best_similarity = 0
            for pathway_dct in pathway_list:
                similarity = pathway_dct['similarity']
                if best_similarity < similarity:
                    best_similarity = similarity
            similarities.append(best_similarity)

        similarities_np = np.array(similarities)
        print(similarities_np.shape)
        zeros = (similarities_np == 0).sum()
        print(f"There are {zeros} smiles with a similarity of 0")
        ones = (similarities_np == 1).sum()
        print(f"There are {ones} smiles with a similarity of 1")

        bins = np.arange(0.0, 1.01, 0.1)
        similarity_bins = np.digitize(similarities_np, bins, right=False)
        counts = np.bincount(similarity_bins, minlength=len(bins) + 1)
        print(counts[1:])

    """
    Start with embeddings_100000_557153.pth
    (100001,)
    There are 5334 smiles with a similarity of 0
    There are 21419 smiles with a similarity of 1
    [ 5380   592  3104  7717 12147 14950 13758 12000  6997  1937 21419]
    Start with embeddings_105000_568145.pth
    (105001,)
    There are 5597 smiles with a similarity of 0
    There are 22402 smiles with a similarity of 1
    [ 5647   611  3231  8125 12791 15755 14469 12592  7342  2036 22402]
    """
    np.set_printoptions()


def filter_pathways():
    total = successful = 0

    pathway_files = os.listdir(PATHWAY_PATH)
    pathway_files = [pf for pf in pathway_files if pf[:10] == "embeddings"]
    filtered_dct = {}
    for pathway_file in pathway_files:
        dct = torch.load(os.path.join(PATHWAY_PATH, pathway_file))
        for smiles, pathway_list in dct.items():
            # print(f"\n{smiles}")

            pathway = None
            for pathway_dct in pathway_list:
                similarity = pathway_dct['similarity']
                if similarity == 1:
                    make_pathway(pathway_dct)
                    analog_smiles = pathway_dct['analog']
                    if analog_smiles == smiles:
                        filtered_dct[smiles] = make_pathway(pathway_dct)
                        successful += 1
                        # print(filtered_dct[smiles])
                        pathway = None
                        break
                    else:
                        if pathway is None:
                            pathway = make_pathway(pathway_dct)
            if pathway is not None:
                filtered_dct[smiles] = pathway
                successful += 1
                # print(analog_smiles)
                # print(filtered_dct[smiles])

            total += 1
        print(successful, total)
    torch.save(filtered_dct, FILTERED_SMILES_PATH)


def make_pathway(pathway_dct):
    token_types = pathway_dct['types']
    reactant_indices = pathway_dct['reactant']
    rxn_indices = pathway_dct['rxn']

    pathway = []
    for token_type, building_block, reaction in zip(token_types, reactant_indices, rxn_indices):
        if token_type == 1:
            pathway.append((int(token_type), -1))
        elif token_type == 3:
            pathway.append((int(token_type), int(building_block)))
        elif token_type == 2:
            pathway.append((int(token_type), int(reaction)))
        elif token_type == 0:
            pathway.append((int(token_type), -1))
            break
        else:
            print(f"Token type {token_type} does not exist")
            sys.exit(1)

    return pathway


def filter_proteins():
    # Get target_id's of smiles for which we have a pathway
    pap_smiles = pd.read_csv(PAPYRUS_SMILES, sep="\t")
    smiles_dct = torch.load(FILTERED_SMILES_PATH)
    smiles = smiles_dct.keys()
    pap_smiles = pap_smiles[pap_smiles["SMILES"].isin(smiles)]
    print(pap_smiles.shape)
    """125411"""
    target_ids = pap_smiles["target_id"].unique()
    print(target_ids.shape)
    """2590"""

    # Filter and store the new protein dict
    protein_dct = torch.load(FILTERED_PROTEIN_PATH1)
    print(len(protein_dct))
    """5074"""
    selected_protein_dct = {k: v.half() for k, v in protein_dct.items() if (k in target_ids) and (v.shape[0] <= 2002)}
    print(len(selected_protein_dct))
    """2537"""
    torch.save(selected_protein_dct, FILTERED_PROTEIN_PATH2)

    # Analyse protein length distribution
    pap_protein = pd.read_csv(PAPYRUS_PROTEIN, sep="\t")
    pap_protein = pap_protein[pap_protein["target_id"].isin(target_ids)]
    print(pap_protein["Length"].min(), pap_protein["Length"].max())
    print((pap_protein["Length"] > 2000).sum())  # 53
    print((pap_protein["Length"] > 3000).sum())  # 14
    print((pap_protein["Length"] > 4000).sum())  # 7
    print((pap_protein["Length"] > 5000).sum())  # 2
    print((pap_protein["Length"] > 7000).sum())  # 2
    # analyse_protein_distribution(pap_protein)


def filter_papyrus():
    papyrus = pd.read_csv(PAPYRUS_SMILES, sep="\t")
    papyrus = papyrus[['SMILES', 'target_id']]

    smiles_dct = torch.load(FILTERED_SMILES_PATH)
    print("#molecules", len(smiles_dct))
    smiles = smiles_dct.keys()
    papyrus = papyrus[papyrus["SMILES"].isin(smiles)]

    protein_dct = torch.load(FILTERED_PROTEIN_PATH2)
    print("#proteins ", len(protein_dct))
    target_ids = protein_dct.keys()
    papyrus = papyrus[papyrus["target_id"].isin(target_ids)]

    print("Papyrus selection shape:", papyrus.shape)
    print("#molecules", len(papyrus['SMILES'].unique()))
    print("#proteins ", len(papyrus['target_id'].unique()))
    papyrus.to_csv(PAPYRUS_FILTERED, index=False)
    """
    Papyrus selection shape: (123236, 2)
    #molecules 22383
    #proteins  2537
    """


def load_papyrus_processed():
    smiles_dct = torch.load(FILTERED_SMILES_PATH)
    protein_dct = torch.load(FILTERED_PROTEIN_PATH2)
    protein_dct = {k: v.cpu().numpy() for k, v in protein_dct.items()}

    papyrus = pd.read_csv(PAPYRUS_FILTERED)
    papyrus['pathway'] = papyrus['SMILES'].map(smiles_dct)
    papyrus['protein_embedding'] = papyrus['target_id'].map(protein_dct)
    papyrus = papyrus.drop(columns=['SMILES', 'target_id'])
    return papyrus


if __name__ == "__main__":
    run()
