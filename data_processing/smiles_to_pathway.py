import time

import torch
# from omegaconf import OmegaConf

# My own imports:
import pandas as pd
from synformer.scripts.sample_naive import load_model, featurize_smiles
# from synformer.models.synformer import draw_generation_results

from data_processing.papyrus_eda import get_smiles

MODEL_PATH = "../data/trained_weights/sf_ed_default.ckpt"
CONFIG_PATH = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# REPEAT = 4
# BATCH_SIZE = 5000
# MAX_LENGTH = 16
OFFSET = 270000

def run():
    model, fpindex, rxn_matrix = load_model(MODEL_PATH, CONFIG_PATH, DEVICE)
    smiles_series = get_smiles()

    output_data = {}
    start_time = intermediate_time = time.time()
    for i, row in smiles_series.iterrows():
        if i <= OFFSET:
            continue

        smiles, count, cumsum = row['SMILES'], row['counts'], row['cumulative']
        # Occurs in 256540 C[C+](=O)(Nc1ccccc1)C(C#N)=Cc1cc(O)c(O)cc1 which causes an error in Synformer
        if 'C+' in smiles:
            continue

        if count >= 10:
            max_length, repeat, batch_size = 32, 16, 2500
        elif count >= 2:
            max_length, repeat, batch_size = 24, 8, 5000
        else:
            max_length, repeat, batch_size = 16, 4, 10000

        print(i, smiles)
        output_i = get_synthetic_pathway(model, fpindex, rxn_matrix, smiles, max_length=max_length, repeat=repeat)
        print(f"Pathways found: {len(output_i)}, Similarities: {[outp['similarity'] for outp in output_i]}")
        output_data[smiles] = output_i

        if (i != 0) and (i%batch_size == 0):
            torch.save(output_data, f"data/synthetic_pathways/embeddings_{i:06d}_{cumsum}.pth")
            output_data = {}

            end_time = time.time()
            print(f"Total time: {end_time - start_time}, batch time: {end_time - intermediate_time}")
            intermediate_time = end_time
    torch.save(output_data, f"data/synthetic_pathways/embeddings_{i:06d}.pth")


def get_synthetic_pathway(model, fpindex, rxn_matrix, smiles, max_length=24, repeat=1):
    mol, feat = featurize_smiles(smiles, DEVICE, repeat=repeat)
    with torch.inference_mode():
        result = model.generate_without_stack(
            feat,
            rxn_matrix=rxn_matrix,
            fpindex=fpindex,
            max_len=max_length,
            temperature_token=1.0,
            temperature_reactant=0.1,
            temperature_reaction=1.0,
        )
    stacks = result.build()

    output_i = []
    for j, stack in enumerate(stacks):
        # Only those sequences with stack depth of 1 (i.e. applying the building blocks and reactions leads to 1 final molecule)
        # are considered valid results!!
        if stack.get_stack_depth() == 1:
            smiles_analog = stack.get_one_top()

            output_ij = {
                'analog': smiles_analog.smiles,
                'similarity': smiles_analog.sim(mol),

                'padding': result.token_padding_mask[j],
                'types': result.token_types[j],
                'reactant': result.reactant_indices[j],
                'rxn': result.rxn_indices[j],

                'reactant_fps': compress_sparse_matrix(result.reactant_fps[j]),
                'predicted_fps': compress_sparse_matrix(result.predicted_fps[j])
            }
            output_i.append(output_ij)

    return output_i


def compress_sparse_matrix(tensor):
    non_zero_indices = (tensor == 1).nonzero(as_tuple=False)
    return non_zero_indices


if __name__ == "__main__":
    # run_Janiks_code()
    run()
