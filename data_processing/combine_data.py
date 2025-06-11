import os
import sys

import numpy as np
import pandas as pd
import torch

PATHWAY_PATH = r"../data/synthetic_pathways"
FILTERED_SMILES_PATH1 = os.path.join(PATHWAY_PATH, "filtered_pathways.pth")
FILTERED_SMILES_PATH2 = os.path.join(PATHWAY_PATH, "filtered_pathways_370000.pth")
# FILTERED_SMILES_PATH2 = os.path.join(PATHWAY_PATH, "filtered_pathways_7.pth")
# FILTERED_SMILES_PATH2 = os.path.join(PATHWAY_PATH, "filtered_pathways_5.pth")

PROTEIN_PATH = r"../data/protein_embeddings"
FILTERED_PROTEIN_PATH1 = os.path.join(PROTEIN_PATH, "usefull_embeddings.pth")
FILTERED_PROTEIN_PATH2 = os.path.join(PROTEIN_PATH, "embeddings_selection_float16.pth")

PAPYRUS_PATH = r"../data/papyrus"
PAPYRUS_SMILES = os.path.join(PAPYRUS_PATH, "05.6++_combined_set_without_stereochemistry.tsv.xz")
PAPYRUS_PROTEIN = os.path.join(PAPYRUS_PATH, "05.6_combined_set_protein_targets.tsv.xz")
PAPYRUS_FILTERED = os.path.join(PAPYRUS_PATH, "papyrus_selection.csv")
LOAD_PAPYRUS_FILTERED = os.path.join(PAPYRUS_PATH, "papyrus_selection_195372.csv")

def run():
    analyse_pathway_similarity()

    filter_pathways()
    # filter_pathways(th=0.7)
    # filter_pathways(th=0.5)

    filter_proteins()

    filter_papyrus()

    # Load the data as a test
    papyrus = load_papyrus_processed()
    print(papyrus)
    print(papyrus.shape)


def analyse_pathway_similarity():
    """
    Goes over the generated embeddings and calculates statistics about the molecule similarities.
    An example output is given at the bottom
    """
    np.set_printoptions(threshold=np.inf)

    pathway_files = os.listdir(PATHWAY_PATH)
    pathway_files = [pf for pf in pathway_files if pf[:10] == "embeddings"]
    similarities = []
    num_pathways = []
    hit_ids = []
    for pathway_file in pathway_files:
        print(f"Start with {pathway_file}")
        dct = torch.load(os.path.join(PATHWAY_PATH, pathway_file))

        for smiles, pathway_list in dct.items():
            best_similarity, hit_id = 0, -1
            for i, pathway_dct in enumerate(pathway_list):
                similarity = pathway_dct['similarity']
                if best_similarity < similarity:
                    best_similarity = similarity
                    if similarity == 1:
                        hit_id = i
            similarities.append(best_similarity)
            num_pathways.append(len(pathway_list))
            hit_ids.append(hit_id)

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

    num_tries = np.zeros(len(similarities))
    num_tries[:5080] = 16
    num_tries[5080:234988] = 8
    num_tries[234988:] = 4
    data = {
        'similarities': similarities,
        'num_tries': num_tries,
        'num_pathways': num_pathways,
        'hit_ids': hit_ids
    }
    df = pd.DataFrame(data)
    df.to_csv("pathway_stats.csv", index=False)

    """
    (369999,)
    There are 29969 smiles with a similarity of 0
    There are 70936 smiles with a similarity of 1
    [30166  2302 11545 29032 45070 56150 50223 42990 24869  6716 70936]
    """

    # Reset the print options
    np.set_printoptions()


def filter_pathways(th=1):
    """
    Selects and stores the pathways that have a similarity of at least th.
    If th is equal to 1 it means that the pathways produce a molecule that is an exact match.
    """
    total = successful = 0

    pathway_files = os.listdir(PATHWAY_PATH)
    pathway_files = [pf for pf in pathway_files if pf[:10] == "embeddings"]
    filtered_dct = {}
    for pathway_file in pathway_files:
        dct = torch.load(os.path.join(PATHWAY_PATH, pathway_file))
        for smiles, pathway_list in dct.items():
            # print(f"\n{smiles}")

            best_pathway, best_similarity = None, 0
            for pathway_dct in pathway_list:
                curr_similarity = pathway_dct['similarity']
                if curr_similarity >= th:
                    curr_pathway = make_pathway(pathway_dct)
                    if curr_similarity == 1:
                        analog_smiles = pathway_dct['analog']
                        if analog_smiles == smiles:
                            filtered_dct[smiles] = curr_pathway
                            successful += 1
                            # print(filtered_dct[smiles])
                            best_pathway = None
                            break

                    if curr_similarity > best_similarity:
                        best_pathway, best_similarity = curr_pathway, curr_similarity

            if best_pathway is not None:
                filtered_dct[smiles] = best_pathway
                successful += 1
                # print(analog_smiles)
                # print(filtered_dct[smiles])

            total += 1
        print(successful, total)
    torch.save(filtered_dct, os.path.join(PATHWAY_PATH, f"filtered_pathways_{int(th*10)}.pth"))


def make_pathway(pathway_dct):
    """
    Generates a more compact representation of a pathway
    """
    token_types = pathway_dct['types']
    reactant_indices = pathway_dct['reactant']
    rxn_indices = pathway_dct['rxn']

    pathway = []
    for token_type, building_block, reaction in zip(token_types, reactant_indices, rxn_indices):
        if token_type == 1:  # Start token
            pathway.append((int(token_type), -1))
        elif token_type == 3:  # Building block token
            pathway.append((int(token_type), int(building_block)))
        elif token_type == 2:  # Reaction token
            pathway.append((int(token_type), int(reaction)))
        elif token_type == 0:  # End token
            pathway.append((int(token_type), -1))
            break
        else:
            print(f"Token type {token_type} does not exist")
            sys.exit(1)

    return pathway


def filter_proteins():
    # Get target_id's of smiles for which we have a pathway
    pap_smiles = pd.read_csv(PAPYRUS_SMILES, sep="\t")
    smiles_dct = torch.load(FILTERED_SMILES_PATH2)
    smiles = smiles_dct.keys()
    pap_smiles = pap_smiles[pap_smiles["SMILES"].isin(smiles)]
    print(pap_smiles.shape)
    """125411 - 185792 - 199422"""
    target_ids = pap_smiles["target_id"].unique()
    print(target_ids.shape)
    """2590 - 3006 - 3068"""

    # Filter and store the new protein dict
    protein_dct = torch.load(FILTERED_PROTEIN_PATH1)
    print(len(protein_dct))
    """5074"""
    # The line below filters the proteins by length and convert the embeddings from float32 to float16
    selected_protein_dct = {k: v.half() for k, v in protein_dct.items() if (k in target_ids) and (v.shape[0] <= 2002)}  # (v.shape[0] <= 2002)}
    print(len(selected_protein_dct))
    """2537 - 2945 - 3004 -- 4973"""
    torch.save(selected_protein_dct, FILTERED_PROTEIN_PATH2)

    # Analyse protein length distribution
    pap_protein = pd.read_csv(PAPYRUS_PROTEIN, sep="\t")
    pap_protein = pap_protein[pap_protein["target_id"].isin(target_ids)]
    print(pap_protein["Length"].min(), pap_protein["Length"].max())
    print((pap_protein["Length"] > 2000).sum())  # 53 - 61 - 64
    print((pap_protein["Length"] > 3000).sum())  # 14 - 17 - 18
    print((pap_protein["Length"] > 4000).sum())  # 7 - 8
    print((pap_protein["Length"] > 5000).sum())  # 2
    print((pap_protein["Length"] > 7000).sum())  # 2


def filter_papyrus():
    """
    Selects the pairs from Papyrus++ that can be made after the molecule and protein filtering that has been done.
    """
    papyrus = pd.read_csv(PAPYRUS_SMILES, sep="\t")
    papyrus = papyrus[['SMILES', 'target_id']]

    smiles_dct = torch.load(FILTERED_SMILES_PATH2)
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
    # papyrus.to_csv(PAPYRUS_FILTERED, index=False)
    """
    105000 synformer runs
    Papyrus selection shape: (123236, 2)
    #molecules 22383
    #proteins  2537
    
    -
    
    290000 synformer runs
    Papyrus selection shape: (182129, 2)
    #molecules 56991
    #proteins  2945
    
    -
    
    370000 synformer runs
    Papyrus selection shape: (195372, 2)
    #molecules 70234
    #proteins  3004
    
    ---
    
    Selection criteria: similarity >= 0.7
    Papyrus selection shape: (394699, 2)
    #molecules 144578
    #proteins  3769
    
    -
    
    Selection criteria: similarity >= 0.5
    Papyrus selection shape: (673520, 2)
    #molecules 248585
    #proteins  4400
    """


def load_papyrus_processed():
    """
    Loads and combines all the Papyrus, protein embedding and pathways data.
    """
    smiles_dct = torch.load(FILTERED_SMILES_PATH2)
    protein_dct = torch.load(FILTERED_PROTEIN_PATH2)
    protein_dct = {k: v.cpu().numpy() for k, v in protein_dct.items()}

    papyrus = pd.read_csv(LOAD_PAPYRUS_FILTERED)
    papyrus['pathway'] = papyrus['SMILES'].map(smiles_dct)
    papyrus['protein_embedding'] = papyrus['target_id'].map(protein_dct)
    papyrus = papyrus.drop(columns=['SMILES', 'target_id'])
    return papyrus


if __name__ == "__main__":
    run()
