import time

import torch

import pandas as pd
from synformer.scripts.sample_naive import load_model, featurize_smiles

MODEL_PATH = "../data/trained_weights/sf_ed_default.ckpt"
CONFIG_PATH = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Continue from this index
OFFSET = 370000

def run():
    # Load the model and the data
    model, fpindex, rxn_matrix = load_model(MODEL_PATH, CONFIG_PATH, DEVICE)
    smiles_series = get_smiles()

    output_data = {}
    start_time = intermediate_time = time.time()
    for i, row in smiles_series.iterrows():
        if i <= OFFSET:
            continue

        smiles, count, cumsum = row['SMILES'], row['counts'], row['cumulative']
        # There was one molecule of which the SMILES caused the program to crash. Hence, it was skipped
        # Occurs in 256540 C[C+](=O)(Nc1ccccc1)C(C#N)=Cc1cc(O)c(O)cc1 which causes an error in Synformer
        if 'C+' in smiles:
            continue

        # More computational resources and storage where reserved for molecules that have
        if count >= 10:
            max_length, repeat, batch_size = 32, 16, 2500
        elif count >= 2:
            max_length, repeat, batch_size = 24,  8, 5000
        else:
            max_length, repeat, batch_size = 16,  4, 10000

        # Compute pathway and store the information in the dictionary
        print(i, smiles)
        output_i = get_synthetic_pathway(model, fpindex, rxn_matrix, smiles, max_length=max_length, repeat=repeat)
        print(f"Pathways found: {len(output_i)}, Similarities: {[outp['similarity'] for outp in output_i]}")
        output_data[smiles] = output_i

        # Periodically store the intermediate results
        if (i != 0) and (i%batch_size == 0):
            torch.save(output_data, f"data/synthetic_pathways/embeddings_{i:06d}_{cumsum}.pth")
            output_data = {}

            end_time = time.time()
            print(f"Total time: {end_time - start_time}, batch time: {end_time - intermediate_time}")
            intermediate_time = end_time
    # Store the final results
    torch.save(output_data, f"data/synthetic_pathways/embeddings_{i:06d}_{cumsum}.pth")


def get_smiles():
    """Loads the smiles from Papyrus++"""
    df = pd.read_csv(r"../data/papyrus/05.6++_combined_set_without_stereochemistry.tsv.xz", sep="\t")
    val_counts = df['SMILES'].value_counts()  # .value_counts().sort_index(ascending=False)

    val_counts = pd.DataFrame({
        'SMILES': val_counts.index,
        'counts': val_counts.values,
        'cumulative': val_counts.values.cumsum()
    })  # .reset_index()
    pd.set_option('display.max_rows', None)
    print(val_counts.iloc[[0, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 75000, 100000, 150000, 250000, 500000, -1]])
    pd.reset_option('display.max_rows')
    """               Count  Cumsum count
    Length >=    10     782          5080
    Length >=     2  133995        234988
    Length >=     1  471174        706162
    """
    return val_counts


def get_synthetic_pathway(model, fpindex, rxn_matrix, smiles, max_length=24, repeat=1):
    # Run the model
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

    # Check for useable pathways and format the output for storage
    output_i = []
    for j, stack in enumerate(stacks):
        # Only those sequences with stack depth of 1 (i.e. applying the building blocks and
        # reactions leads to 1 final molecule) are considered valid results!!
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
    """Converts a sparse matrix with a few ones and many zeros to a list of indices of the ones."""
    non_zero_indices = (tensor == 1).nonzero(as_tuple=False)
    return non_zero_indices


if __name__ == "__main__":
    run()
