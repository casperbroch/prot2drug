import os

import pandas as pd
import torch
import pulp

from data_processing.combine_data import PATHWAY_PATH, PAPYRUS_PATH

LOAD_PAPYRUS_FILTERED, SELECT_NUM_PROT = os.path.join(PAPYRUS_PATH, "papyrus_selection_195372.csv"), 300
# LOAD_PAPYRUS_FILTERED, SELECT_NUM_PROT = os.path.join(PAPYRUS_PATH, "papyrus_selection_7_394699.csv"), 377
# LOAD_PAPYRUS_FILTERED, SELECT_NUM_PROT = os.path.join(PAPYRUS_PATH, "papyrus_selection_5_673520.csv"), 440

FILTERED_SMILES_PATH2 = os.path.join(PATHWAY_PATH, "filtered_pathways_5.pth")



def run():
    # data_split_additive_weights()
    data_split_protein_first()


def data_split_additive_weights():
    """
    Naive approach of splitting the data based on scores assigned to pairs based on how often their protein and
    molecule occure in other pairs.
    """
    # Load the data and print some stats
    papyrus = pd.read_csv(LOAD_PAPYRUS_FILTERED)
    print(papyrus.columns)
    print(papyrus['SMILES'].value_counts())
    print(papyrus['target_id'].value_counts())

    smiles_dct = torch.load(FILTERED_SMILES_PATH2)
    papyrus['pathways'] = papyrus['SMILES'].map(smiles_dct)
    print(papyrus['pathways'].apply(len).value_counts().sort_index())

    # Remove pathways longer than 9
    papyrus = papyrus[papyrus['pathways'].apply(len)<=9]

    # Calculate the weights
    papyrus['SMILES_weight'] = papyrus['SMILES'].map(papyrus['SMILES'].value_counts())
    papyrus['prot_weight'] = papyrus['target_id'].map(papyrus['target_id'].value_counts())

    papyrus['pair_weight'] = papyrus['SMILES_weight'] + papyrus['prot_weight']
    weighted_counts = papyrus['pair_weight'].value_counts().sort_index().to_frame(name='count')
    weighted_counts['cumsum'] = weighted_counts.cumsum()
    weighted_counts['pct'] = weighted_counts['cumsum'] / weighted_counts['cumsum'].max() * 100
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(weighted_counts)
        print(weighted_counts[weighted_counts > 1].shape)

    # Split the data
    test_cutoff = weighted_counts[weighted_counts['pct'] > 10]['pct'].idxmin()
    val_cutoff = weighted_counts[weighted_counts['pct'] > 20]['pct'].idxmin()
    total_pairs = len(papyrus)
    print(test_cutoff, val_cutoff, total_pairs)
    papyrus_test = papyrus[papyrus['pair_weight'] <= test_cutoff]
    papyrus_val = papyrus[(test_cutoff < papyrus['pair_weight']) & (papyrus['pair_weight'] <= val_cutoff)]
    papyrus_train = papyrus[papyrus['pair_weight'] > val_cutoff]

    # Analyse the contamination between sets
    cross_contamination_analysis(papyrus_train, papyrus_val, papyrus_test, total_pairs)


def data_split_protein_first():
    """
    Split the data by selecting all pairs of a subset of proteins.
    Uses and ILP to choose the proteins.
    """
    # Load the data and print some stats
    papyrus = pd.read_csv(LOAD_PAPYRUS_FILTERED)
    print(papyrus.columns)
    print(papyrus['SMILES'].value_counts())
    print(papyrus['target_id'].value_counts())

    smiles_dct = torch.load(FILTERED_SMILES_PATH2)
    papyrus['pathways'] = papyrus['SMILES'].map(smiles_dct)
    print(pd.Series(papyrus['SMILES'].unique()).map(smiles_dct).apply(len).value_counts().sort_index())
    print(papyrus['SMILES'].unique().shape)
    print(papyrus['pathways'].apply(len).value_counts().sort_index())
    print(len(papyrus['pathways']))

    # Remove pathways longer than 9
    papyrus = papyrus[papyrus['pathways'].apply(len)<=9]

    papyrus['SMILES_weight'] = papyrus['SMILES'].map(papyrus['SMILES'].value_counts())
    papyrus['prot_weight'] = papyrus['target_id'].map(papyrus['target_id'].value_counts())

    # Start splitting of the data
    papyrus_test, papyrus_val_train = get_subset(papyrus, 10)
    papyrus_val, papyrus_train = get_subset(papyrus_val_train, 10/(90/100))

    # Analyse results
    total_pairs = len(papyrus)
    cross_contamination_analysis(papyrus_train, papyrus_val, papyrus_test, total_pairs)

    # Process and save datasets
    papyrus_test = papyrus_test[['SMILES', 'target_id']]
    papyrus_test_len = len(papyrus_test)
    papyrus_val = papyrus_val[['SMILES', 'target_id']]
    papyrus_val_len = len(papyrus_val)
    papyrus_train = papyrus_train[['SMILES', 'target_id']]
    papyrus_train_len = len(papyrus_train)

    test_path = os.path.join(PAPYRUS_PATH, f"papyrus_test_{papyrus_test_len}.csv")
    val_path = os.path.join(PAPYRUS_PATH, f"papyrus_val_{papyrus_val_len}.csv")
    train_path = os.path.join(PAPYRUS_PATH, f"papyrus_train_{papyrus_train_len}.csv")
    papyrus_test.to_csv(test_path, index=False)
    papyrus_val.to_csv(val_path, index=False)
    papyrus_train.to_csv(train_path, index=False)


def get_subset(df, select_pct):
    """
    Uses an ILP to split the df into two sets. The smallest set is select_pct percenatage of the total df.
    """
    weighted_counts = df['target_id'].value_counts().sort_values().to_frame(name='count').reset_index(
        inplace=False)
    weighted_counts['cumsum'] = weighted_counts['count'].cumsum()
    weighted_counts['count_pct'] = weighted_counts['count'] / weighted_counts['cumsum'].max() * 100
    weighted_counts['pct'] = weighted_counts['cumsum'] / weighted_counts['cumsum'].max() * 100

    def return_smiles_count(papyrus_df):
        def smiles_count(target_id):
            smiles = papyrus_df.loc[papyrus_df['target_id'] == target_id, 'SMILES'].unique()
            return papyrus_df['SMILES'].isin(smiles).sum()

        return smiles_count

    weighted_counts['avg_smiles_occurences'] = weighted_counts['target_id'].apply(return_smiles_count(df))
    weighted_counts['avg_smiles_occurences'] = weighted_counts['avg_smiles_occurences'] / weighted_counts['count']
    weighted_counts = weighted_counts.sort_values(by='avg_smiles_occurences')
    subset_target_ids = ilp_hard_constraint(weighted_counts, target_pct=select_pct)
    papyrus_subset = df[df['target_id'].isin(subset_target_ids)]
    papyrus_remainder = df[~df['target_id'].isin(subset_target_ids)]

    weighted_counts_subset = weighted_counts[weighted_counts['target_id'].isin(subset_target_ids)]
    print(len(weighted_counts_subset))
    print('Total instances', weighted_counts_subset['count'].sum())
    print('Avg smiles', (weighted_counts_subset['avg_smiles_occurences'] * weighted_counts_subset['count']).sum())
    used_smiles = df.loc[df['target_id'].isin(subset_target_ids), 'SMILES'].unique()
    print('Total smiles', len(used_smiles))
    print('SMILES contamination', df['SMILES'].isin(used_smiles).sum())

    return papyrus_subset, papyrus_remainder


def ilp_soft_constraint(df, target_pct=10):
    """
    This ILP chooses 300 proteins that form pairs with molecules that on average form the least pairs with other
    proteins. It also optimized for a total number of pairs as close to 10% as possible.
    """
    n = len(df)
    v1 = df['count'].to_numpy()
    v2 = df['avg_smiles_occurences'].to_numpy()
    lambda_penalty = 10

    target_v1_sum = target_pct/100 * v1.sum()

    # Define the ILP problem
    prob = pulp.LpProblem("SubsetSelection", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

    deviation = pulp.LpVariable("deviation", lowBound=0)
    objective = (
            pulp.lpSum([x[i] * v1[i] * v2[i] for i in range(n)]) +
            lambda_penalty * deviation
    )
    prob += objective
    prob += pulp.lpSum(x) == SELECT_NUM_PROT
    v1_sum_expr = pulp.lpSum([x[i] * v1[i] for i in range(n)])
    prob += (v1_sum_expr - target_v1_sum <= deviation)
    prob += (target_v1_sum - v1_sum_expr <= deviation)

    # Solve it
    prob.solve(pulp.PULP_CBC_CMD(msg=True))

    # Output selected indices
    selected_indices = [i for i in range(n) if pulp.value(x[i]) == 1]
    target_ids = df.iloc[selected_indices, 'target_id']
    return target_ids


def ilp_hard_constraint(df, target_pct=10):
    """
    This ILP chooses 300 proteins that form pairs with molecules that on average form the least pairs with other
    proteins. The total number of pairs should be at least 10% of the total.
    """
    n = len(df)
    v1 = df['count'].to_numpy()
    v2 = df['avg_smiles_occurences'].to_numpy()

    target_v1_sum = target_pct/100 * v1.sum()

    # Define the ILP problem
    prob = pulp.LpProblem("SubsetSelection", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(n)]

    prob += pulp.lpSum([x[i] * v1[i] * v2[i] for i in range(n)])
    prob += pulp.lpSum(x) == SELECT_NUM_PROT
    prob += pulp.lpSum([x[i] * v1[i] for i in range(n)]) >= target_v1_sum

    prob.solve(pulp.PULP_CBC_CMD(msg=True))

    # Output selected indices
    selected_indices = [i for i in range(n) if pulp.value(x[i]) == 1]
    target_ids = df.iloc[selected_indices]['target_id']
    return target_ids



def cross_contamination_analysis(papyrus_train, papyrus_val, papyrus_test, total_pairs):
    """
    Analyses the contamination between all datasets
    """
    print('For test contaminant is val')
    cross_contamination(papyrus_test, papyrus_val, total_pairs)
    print('For test contaminant is train')
    cross_contamination(papyrus_test, papyrus_train, total_pairs)
    print('For test contaminant is both')
    cross_contamination(papyrus_test, pd.concat([papyrus_val, papyrus_train]), total_pairs)

    print('For val contaminant is test')
    cross_contamination(papyrus_val, papyrus_test, total_pairs)
    print('For val contaminant is train')
    cross_contamination(papyrus_val, papyrus_train, total_pairs)
    print('For val contaminant is both')
    cross_contamination(papyrus_val, pd.concat([papyrus_test, papyrus_train]), total_pairs)

    print('For train contaminant is test')
    cross_contamination(papyrus_train, papyrus_test, total_pairs)
    print('For train contaminant is val')
    cross_contamination(papyrus_train, papyrus_val, total_pairs)
    print('For train contaminant is both')
    cross_contamination(papyrus_train, pd.concat([papyrus_test, papyrus_val]), total_pairs)


def cross_contamination(pure_df, contaminant, total_pairs):
    """
    Analyses the contamination between two datasets.
    """
    pure_df = pure_df.copy()
    pure_df['SMILES_double'] = pure_df['SMILES'].isin(contaminant['SMILES'])
    pure_df['prot_double'] = pure_df['target_id'].isin(contaminant['target_id'])
    print(pd.crosstab(pure_df['SMILES_double'], pure_df['prot_double']) / total_pairs)

if __name__ == "__main__":
    run()
