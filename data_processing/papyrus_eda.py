import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def run():
    smiles = get_data(data_type='mol')
    proteins = get_data(data_type='prot')

    # Summarize all columns from Papyrus++ protein and molecule data files
    for df in (smiles, proteins):
        print(df.columns)
        print(df.shape)

        for col in df.columns:
            print()
            print(col)
            print(f"Number of NaN:    {df[col].isna().sum()}")
            uniq = df[col].unique()
            print(f"Number of unique: {len(uniq)}")

            if not col in ('type_IC50', 'type_EC50', 'type_KD', 'type_Ki'):
                print(uniq)
            else:
                lens = [len(un) for un in uniq]
                print(lens)

    targeted_questions(proteins, smiles)

    analyse_both(proteins, smiles)


def targeted_questions(proteins, smiles):
    """Used to calculate specific values for the report"""
    sel_proteins = proteins[proteins["target_id"].isin(smiles['target_id'])]
    print("Max length", sel_proteins["Length"].max())
    print("Average length", sel_proteins["Length"].mean())
    print("Median length", sel_proteins["Length"].median())
    print("Remaining pairs", len(smiles[smiles["target_id"].isin(sel_proteins.loc[sel_proteins['Length'] <= 2000, "target_id"])]))
    print("Number proteins <= 2000 and total", len(sel_proteins.loc[sel_proteins['Length'] <= 2000]), len(sel_proteins))
    print("Sum proteins <= 2000 and total", sel_proteins.loc[sel_proteins['Length'] <= 2000, "Length"].sum(), sel_proteins["Length"].sum())
    print("Squared sum proteins <= 2000 and total", (sel_proteins.loc[sel_proteins['Length'] <= 2000, "Length"]**2).sum(),
          (sel_proteins["Length"]**2).sum())
    print("Num organisms", len(sel_proteins['Organism'].unique()))
    # number of organisms


def analyse_protein_distribution(df):
    """
    Makes the plots from the report that show the protein length distribution
    and the computational complexity plot
    """
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 20})
    capped_data = [min(x, 2005) if x != 2000 else 1999 for x in df["Length"]]
    bins = list(range(0, 2000, 250)) + [2000, 2250]
    plt.hist(capped_data, bins=bins, edgecolor='black')
    labels = [f'{i}–{i + 249}' for i in range(0, 1750, 250)] + ['1750-2000', '2000+']

    plt.xticks([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)], labels, rotation=20)
    plt.xlim(0, 2250)

    plt.grid(axis='y')
    plt.xlabel("Protein sequence length")
    plt.ylabel("Numbers of proteins")
    plt.tight_layout()

    # Lengths threshold plots
    values = df["Length"].dropna().sort_values().values
    thresholds, idx = np.unique(values, return_index=True)

    counts = np.arange(1, len(values) + 1)[idx]
    sums = np.cumsum(values)[idx]
    squared_sums = np.cumsum(values ** 2)[idx]
    counts_norm = counts / counts.max()
    sums_norm = sums / sums.max()
    squared_sums_norm = squared_sums / squared_sums.max()

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 20})
    plt.plot(thresholds, counts_norm*100, label='Number of proteins', color='blue')
    plt.plot(thresholds, sums_norm*100, label='Computation for $O(n)$ operations', color='green')
    plt.plot(thresholds, squared_sums_norm*100, label='Computation for $O(n^2)$ operations', color='red')
    plt.axvline(x=2000, linestyle='--', color='black')

    plt.xlabel('Threshold in sequence length')
    plt.ylabel('Percentage')
    plt.xlim(0, 7150)
    plt.ylim(0, 100.1)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot top 10 organisms
    top_classes = df["Organism"].value_counts().nlargest(9)
    df["Organism_top10"] = df["Organism"].where(df["Organism"].isin(top_classes.index), "Other")
    final_counts = df["Organism_top10"].value_counts()

    final_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    print(len(df["Organism"]))
    print(final_counts)
    plt.grid(axis='y')
    plt.xlabel("Organism")
    plt.ylabel("Count")
    plt.title("Top 10 Organisms (Others Grouped)")
    plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    plt.show()


def analyse_both(proteins, smiles):
    # Check to see which unique values is the best indicator for the number of unique molecules
    print("InChI", len(smiles['InChI'].unique()))
    print("Key", len(smiles['InChIKey'].unique()))
    prot_id = smiles['target_id'].unique()
    print(smiles['SMILES'])
    print(proteins['target_id'].unique().shape)
    print(prot_id.shape)

    # Make the plots
    proteins = proteins[proteins['target_id'].isin(prot_id)]
    analyse_protein_distribution(proteins)

    # Plot top 10 organisms weighted by the number of pairs in Papyrus++
    df = smiles.merge(proteins[['target_id', 'Organism']], left_on='target_id', right_on='target_id', how='left')
    top_classes = df["Organism"].value_counts().nlargest(9)
    df["Organism_top10"] = df["Organism"].where(df["Organism"].isin(top_classes.index), "Other")
    final_counts = df["Organism_top10"].value_counts()

    final_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    print(len(df["Organism"]))
    print(final_counts)
    plt.grid(axis='y')
    plt.xlabel("Organism")
    plt.ylabel("Count")
    plt.title("Top 10 Organisms (Others Grouped)")
    plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    plt.show()


def get_data(data_type='molecule'):
    """Used to load the .tsv.xz files from Papyrus++"""
    if data_type in ('molecule', 'mol'):
        df = pd.read_csv(r"../data/papyrus/05.6++_combined_set_without_stereochemistry.tsv.xz", sep="\t")
        # df = pd.read_csv(r"data/papyrus/05.6_combined_set_without_stereochemistry.tsv.xz", sep="\t")  # Is to big to load
    elif data_type in ('protein', 'prot'):
        df = pd.read_csv(r"../data/papyrus/05.6_combined_set_protein_targets.tsv.xz", sep="\t")
    return df



if __name__ == "__main__":
    run()
