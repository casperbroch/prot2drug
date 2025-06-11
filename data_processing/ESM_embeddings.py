import time

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM


def get_aminoacid_sequences():
    """Reads in the amino acid dequences from Papyrus++"""
    df = pd.read_csv(r"../data/papyrus/05.6_combined_set_protein_targets.tsv.xz", sep="\t")
    df = df.sort_values(by="Length", ascending=True)
    df = df[['Sequence', 'target_id']]
    df = df.dropna(subset=['Sequence'])
    return df


def get_batch_indices(df, max_batch_size):
    """Groups the indices in batches of amino acids chains of equal length."""
    batch_indices = []

    prev_length = -1
    curr_batch_size = 0
    for i, seq in enumerate(df['Sequence']):
        seq_length = len(seq)
        if seq_length != prev_length or curr_batch_size == max_batch_size:
            batch_indices.append(i)
            curr_batch_size = 0
        else:
            curr_batch_size += 1
        prev_length = seq_length

    batch_indices.append(-1)
    return batch_indices


# Chose the size of ESM you want to run
# model_id = "Synthyra/ESMplusplus_small"
model_id = "Synthyra/ESMplusplus_large"

# Set up model
model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
tokenizer = model.tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Moves model to GPU (if available)

# Get amino acids and set parameters
df = get_aminoacid_sequences()
print(len(df.iloc[0]['Sequence']), len(df.iloc[0]['Sequence']))

max_batch_size = 8
batch_indices = get_batch_indices(df, max_batch_size)
start_index = batch_indices[0]

# Initialize the embeddings dictionary and start loop
embedding_dict = {}
start_time = time.time()
for i_batch, end_index in enumerate(batch_indices[batch_indices.index(start_index)+1:]):
    print("Batch", i_batch)
    batch_start_time = time.time()

    # Ensures slicing of the batches work properly for the last batch
    if end_index == -1:
        batch_df = df[start_index:]
    else:
        batch_df = df[start_index:end_index]

    x_ids = (batch_df["target_id"]).values
    x_sequences = batch_df["Sequence"].values
    for i in range(end_index-start_index):
        print(len(x_sequences[i]), x_sequences[i])

    # Run model
    tokenized_sequences = model.tokenizer(
        x_sequences.tolist(),
        padding=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        output = model(**tokenized_sequences)

    # Store embeddings in dict and print intermediate results
    y_embeddings = output.last_hidden_state
    batch_end_time = time.time()
    print(y_embeddings.shape)
    [print(y_embeddings[i].shape) for i in range(end_index-start_index)]
    print(f"Total time:{batch_end_time - start_time:6.3f}, Batch time:{batch_end_time - batch_start_time:6.3f}")
    print()

    for x_id, y_emb in zip(x_ids, y_embeddings):
        embedding_dict[x_id] = y_emb

    # Intermediate save
    if i_batch % 250 == 0:
        torch.save(embedding_dict, f"data/_esm_large/embeddings_{end_index}_{len(x_sequences[i])}.pth")
    start_index = end_index

# Save all embeddings
torch.save(embedding_dict, "data/_esm_large/embeddings.pth")

# Load the stored embeddings as a test
loaded_embeddings = torch.load("data/_esm_large/embeddings.pth")
for i, key in enumerate(loaded_embeddings.keys()):
    if i < 200:
        print(key, loaded_embeddings[key].shape)
