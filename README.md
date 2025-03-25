# prot2drug
We develop an AI method to generate synthesizable small molecules for hard-to-treat proteins using ESM-3 and SynFormer. The goal is diverse, computationally designed drug candidates for rare and complex diseases.

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
