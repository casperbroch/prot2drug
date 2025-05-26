import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def run():
    smiles = get_data(data_type='mol')
    proteins = get_data(data_type='prot')
    # proteins.loc[proteins['Organism'].str.contains('virus', case=False, na=False), ['Organism']] = 'Virus'

    for df in (smiles, proteins):
        print(df.columns)
        print(df.shape)

        for col in df.columns:
            print()
            print(col)
            print(f"Number of NaN:    {df[col].isna().sum()}")
            uniq = df[col].unique()
            print(f"Number of unique: {len(uniq)}")
            """if col == "SMILES":
                lens = np.array([len(un) for un in uniq])
                print(f'[{np.min(lens)}, {np.max(lens)}]')
                plt.hist(lens)
                plt.show()"""

            if not col in ('type_IC50', 'type_EC50', 'type_KD', 'type_Ki'):
                print(uniq)
            else:
                lens = [len(un) for un in uniq]
                print(lens)

    # analyse_protein_distribution(proteins)
    analyse_both(proteins, smiles)


def analyse_protein_distribution(df):
    # Lengths
    # df["Length"].hist(bins=30, edgecolor="black")

    capped_data = [min(x, 2005) for x in df["Length"]]
    bins = list(range(0, 2000, 250)) + [2000, 2250]
    plt.hist(capped_data, bins=bins, edgecolor='black')
    labels = [f'{i}–{i + 249}' for i in range(0, 2000, 250)] + ['2000+']

    plt.xticks([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)], labels, rotation=45)
    plt.xlim(0, 2250)

    plt.grid(axis='y')
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Protein Lengths")
    # plt.show()

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
    plt.plot(thresholds, counts_norm, label='Constant (Count)', color='blue')
    plt.plot(thresholds, sums_norm, label='Linear (Sum)', color='green')
    plt.plot(thresholds, squared_sums_norm, label='Quadratic (Squared Sum)', color='red')

    plt.xlabel('Threshold')
    plt.ylabel('Normalized Value')
    plt.title('Relative computational needs per Length threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Organisms
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
    print("InChI", len(smiles['InChI'].unique()))
    print("Key", len(smiles['InChIKey'].unique()))
    prot_id = smiles['target_id'].unique()
    print(smiles['SMILES'])
    print(proteins['target_id'].unique().shape)
    print(prot_id.shape)
    proteins = proteins[proteins['target_id'].isin(prot_id)]
    analyse_protein_distribution(proteins)
    df = smiles.merge(proteins[['target_id', 'Organism']], left_on='target_id', right_on='target_id', how='left')

    # Organisms
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
    if data_type in ('molecule', 'mol'):
        df = pd.read_csv(r"../data/papyrus/05.6++_combined_set_without_stereochemistry.tsv.xz", sep="\t")
        # df = pd.read_csv(r"data/papyrus/05.6_combined_set_without_stereochemistry.tsv.xz", sep="\t")  # Is to big to load
    elif data_type in ('protein', 'prot'):
        df = pd.read_csv(r"../data/papyrus/05.6_combined_set_protein_targets.tsv.xz", sep="\t")
    return df


def get_aminoacid_sequences():
    df = pd.read_csv(r"../data/papyrus/05.6_combined_set_protein_targets.tsv.xz", sep="\t")
    df = df.sort_values(by="Length", ascending=True)
    print(df['Sequence'].dropna().values)
    print(df['Sequence'].dropna().values.shape)
    print(type(df['Sequence'].dropna().values))
    sequences = df['Sequence'].dropna().tolist()
    return sequences


def get_smiles():
    value_counts = get_data(data_type='mol')['SMILES'].value_counts()

    value_counts = value_counts
    value_counts = pd.DataFrame({
        'SMILES': value_counts.index,
        'counts': value_counts.values,
        'cumulative': value_counts.values.cumsum()
    })  # .reset_index()
    pd.set_option('display.max_rows', None)
    print(value_counts.iloc[[0, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 75000, 100000, 150000, 250000, 500000, -1]])
    pd.reset_option('display.max_rows')

    # smiles = value_counts['SMILES']
    return value_counts


if __name__ == "__main__":
    run()
