# prot2drug

We develop an AI method to generate synthesizable small molecules for hard-to-treat proteins using ESM-3 and SynFormer. The goal is diverse, computationally designed drug candidates for rare and complex diseases.

---
## **Setup**

First steps
1. Create .env file (see .env.template)

Run ESM locally
1. Create HuggingFace account
2. Request access to the ESM-3-open model: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
3. Generate access token (e.g. read-only token): https://huggingface.co/settings/tokens
4. Put your token in the .env file as the value of the HF_TOKEN variable
5. Create virtual environment
6. Install requirements (requirements.txt) via `pip install -r requirements.txt` 
7. See `ESM Local.ipynb`

Run ESM remotely 
1. Create EvolutionaryScale Forge account: https://forge.evolutionaryscale.ai/ (and tick "non-commercial" box)
2. Create an access token: https://forge.evolutionaryscale.ai/console 
3. Put your token in the .env file as the value of the FORGE_TOKEN variable
4. See `ESM Remote.ipynb`

---
## **Data Download and Setup**
To ensure that everything works correctly, you need to download specific data files from two sources and arrange them in the following directory structure.

### Step 1: Download the Files
#### From [Google Drive](https://drive.google.com/drive/folders/1ZfNaDaaabU96JnVVodbTtIBc3VWKXvif?dmr=1&ec=wgc-drive-hero-goto)
Download the following files from: Google Drive Folder

- pdb_sequences.csv
- pdb_protein_ligand_train.p
- pdb_protein_ligand_test.p
- pdb_ids.csv

#### From [Hugging Face](https://huggingface.co/whgao/synformer/tree/main)
Download the following files from: Synformer Repository on Hugging Face

- fpindex.pkl
- matrix.pkl
- sf_ed_default.ckpt

### Step 2: Arrange the Files
After downloading, create the following folder structure and place each file in the correct location:
```
data
├── pdb_sequences.csv
├── pdb_protein_ligand_train.p
├── pdb_protein_ligand_test.p
├── pdb_ids.csv
├── trained_weights
│   └── sf_ed_default.ckpt
└── processed
    └── comp_2048
        ├── fpindex.pkl
        └── matrix.pkl
```
