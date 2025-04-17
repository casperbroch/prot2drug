import pandas as pd
import torch
from transformers import AutoModelForMaskedLM


def get_aminoacid_sequences():
    df = pd.read_csv(r"data/Papyrus/05.6_combined_set_protein_targets.tsv.xz", sep="\t")
    df = df.sort_values(by="Length", ascending=True)
    print(df['Sequence'].dropna().values)
    print(df['Sequence'].dropna().values.shape)
    print(type(df['Sequence'].dropna().values))
    sequences = df['Sequence'].dropna().tolist()
    return sequences

# model_id = "Synthyra/ESMplusplus_small"
model_id = "Synthyra/ESMplusplus_large"

model = AutoModelForMaskedLM.from_pretrained("Synthyra/ESMplusplus_small", trust_remote_code=True)
tokenizer = model.tokenizer

"""pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"""
print(next(model.parameters()).device)
print(torch.cuda.is_available())  # True if GPU is available
print(torch.cuda.current_device())  # Usually 0 if GPU is in use
print(torch.cuda.get_device_name(0))  # Name of your GPU


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Moves model to GPU (if available)

print()
print(next(model.parameters()).device)
print(torch.cuda.is_available())  # True if GPU is available
print(torch.cuda.current_device())  # Usually 0 if GPU is in use
print(torch.cuda.get_device_name(0))  # Name of your GPU


# Example protein sequences
# start_indx = 0
sequences = get_aminoacid_sequences()  # [start_indx:start_indx+32]
print(len(sequences[0]), len(sequences[-1]))

embedding_dict = model.embed_dataset(
    sequences=sequences,
    tokenizer=model.tokenizer,
    batch_size=4, # adjust for your GPU memory
    max_len=7097, # adjust for your needs  # 1 higher than the maximum in the dataset
    full_embeddings=False, # if True, no pooling is performed
    embed_dtype=torch.float32, # cast to what dtype you want
    pooling_types=['mean', 'cls'], # more than one pooling type will be concatenated together
    num_workers=0, # if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
    sql=False, # if True, embeddings will be stored in SQLite database
    sql_db_path='embeddings.db',
    save=True, # if True, embeddings will be saved as a .pth file
    save_path='embeddings.pth',
)

loaded_embedding = torch.load("embeddings.pth")

for i, seq in enumerate(sequences):
    pass
    print(i, len(seq), seq)
    print(loaded_embedding[seq].shape, loaded_embedding[seq])
    # print(loaded_embedding[seq].shape)
