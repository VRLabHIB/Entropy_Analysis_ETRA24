import os
import glob
import numpy as np
import pandas as pd

import networkx as nx
from sklearn.preprocessing import normalize

from datetime import date

today = date.today()


def locate_transition_data(project_path, datasets):
    if datasets == 'big':
        data_path = project_path + '\\data\\big\\'
    if datasets == 'vir':
        data_path = project_path + '\\data\\vir\\'

    os.chdir(data_path)
    lst = glob.glob("*_trans.csv")

    df_lst = pd.DataFrame({'name': lst})
    df_lst['ID'] = df_lst['name'].str.split('.', expand=True).iloc[:, 0]

    return df_lst


def locate_node_data(project_path, datasets):
    if datasets == 'big':
        data_path = project_path + '\\data\\big\\'
    if datasets == 'vir':
        data_path = project_path + '\\data\\vir\\'

    os.chdir(data_path)
    lst = glob.glob("*_node.csv")

    df_lst = pd.DataFrame({'name': lst})
    df_lst['ID'] = df_lst['name'].str.split('.', expand=True).iloc[:, 0]

    return df_lst


def sum_weights(dft):
    matrix = dft[['Source', 'Target']].to_numpy()

    dft.insert(2, 'combinations', ['_'.join(tupel) for tupel in matrix])

    dft = dft.iloc[:, 2:].groupby('combinations').sum().reset_index()
    dfsplit = dft['combinations'].str.split('_', expand=True)
    dfsplit.columns = ['Source', 'Target']
    dft = dft.drop(columns=['combinations'])
    dft['Source'], dft['Target'] = dfsplit['Source'], dfsplit['Target']
    dft = dft.drop(columns=['start_transition'])

    return dft


def fill_missing_nodes(dft, aoi_lst, item):
    if item == 'edges':
        tupels = dft[['Source', 'Target']].to_numpy()
        for source in aoi_lst:
            for target in aoi_lst:
                if source != target:
                    if [source, target] not in tupels.tolist():
                        zeros = [0] * 5
                        dft.loc[len(dft)] = [source, target] + zeros

    if item == 'nodes':
        for target in aoi_lst:
            if target not in dft['Target'].values:
                zeros = [0] * 3
                dft.loc[len(dft)] = [target] + zeros

    return dft


def calculate_entropies(P, pi):
    # Calculate Ht
    logP = np.nan_to_num(np.log(P.astype(np.float64)))
    sumlogP = np.nan_to_num(P * logP).sum(axis=1)

    Ht = -((pi*sumlogP).sum())

    # Calculate Hs
    logpi = np.nan_to_num(np.log(pi.astype(np.float64)))
    Hs = -((pi*logpi).sum())

    return Ht, Hs


def calculate_entropy_measures(project_path, AOI_lst, datasets):
    data_path = project_path + '\\data\\'

    trans_lst = locate_transition_data(project_path, datasets)
    node_lst = locate_node_data(project_path, datasets)

    # use stepsize of 10sec with 30 sec intervals (20sec overlap)
    length = 10
    if datasets == 'big':
        interval_starts = np.arange(0, 850, length)
    if datasets == 'vir':
        interval_starts = np.arange(0, 600, length)

    dataset_lst = list()
    identifier_lst = list()
    condition_lst = list()
    start_lst = list()
    ht_lst = list()
    hs_lst = list()

    for i in range(0, len(trans_lst)):
        print('index', i)
        dfT = pd.read_csv(project_path + '\\data\\{}\\{}'.format(datasets, trans_lst.iloc[i, 0]))
        dfN = pd.read_csv(project_path + '\\data\\{}\\{}'.format(datasets, node_lst.iloc[i, 0]))

        # Apply lower and upper threshold for AOI duration and transition duration
        dfT = dfT[dfT['trans_duration'] < 4.5] # 99-quantile
        dfN = dfN[dfN['AOI_duration'] > 0.05] # 0.01-quantile

        identifier = dfT['ID'].iloc[0]
        print('ID', identifier)
        condition = dfT['Condition'].iloc[0]

        if identifier != dfN['ID'].iloc[0]:
            raise Exception('Node and transition dataframe have not the same identifier.')

        for start in interval_starts:
            print('int', start)
            dfTs = dfT[np.logical_and(dfT['start_transition'] >= start, dfT['start_transition'] < start + 30)]
            dfNs = dfN[np.logical_and(dfN['duration_start'] >= start, dfN['duration_start'] < start + 30)]

            if np.logical_or(len(dfTs) != 0, len(dfNs) != 0):
                # Create transition matrix
                if len(dfTs) != 0:
                    dfTs = sum_weights(dfTs)
                    dfTs = fill_missing_nodes(dfTs, AOI_lst, 'edges')

                    G = nx.from_pandas_edgelist(dfTs, source='Source', target='Target', edge_attr=['Weight'],
                                                create_using=nx.DiGraph())
                    P = nx.to_numpy_array(G, nodelist=AOI_lst, weight='Weight')

                if len(dfTs) == 0:
                    if datasets == 'big':
                        P = np.zeros((26, 26))
                    if datasets == 'vir':
                        P = np.zeros((19, 19))

                dfNs = dfNs.drop(columns=['ID', 'Condition'])

                dfNs = dfNs.groupby('Target').agg(
                    {"AOI_duration": "sum", "pupil_diameter": "mean", "distance_to_aoi": "mean"}).reset_index()
                dfNs = fill_missing_nodes(dfNs, AOI_lst, 'nodes')
                dfNs['Target'] = dfNs['Target'].astype("category")
                dfNs['Target'] = dfNs['Target'].cat.set_categories(AOI_lst)
                dfNs = dfNs.sort_values(["Target"]).reset_index(drop=True)

                pi = dfNs['AOI_duration'].to_numpy()

                # Normalisation
                P_norm = normalize(P, axis=1, norm='l1')
                pi_norm = pi / pi.sum()

                # Entropy calculation
                Ht, Hs = calculate_entropies(P_norm, pi_norm)

                # collect the data
                dataset_lst.append(datasets)
                identifier_lst.append(identifier)
                condition_lst.append(condition)
                start_lst.append(start)
                ht_lst.append(Ht)
                hs_lst.append(Hs)

            if np.logical_and(len(dfTs) == 0, len(dfNs) == 0):
                # collect the data
                dataset_lst.append(datasets)
                identifier_lst.append(identifier)
                condition_lst.append(condition)
                start_lst.append(start)
                ht_lst.append(0)
                hs_lst.append(0)

        df = pd.DataFrame({'Dataset': dataset_lst, 'ID': identifier_lst, 'Condition': condition_lst, 'Start': start_lst,
                           'transition_entropy': ht_lst, 'stationary_entropy': hs_lst})

        df.to_csv(data_path + '{}_{}_entropy.csv'.format(today, datasets), index=False)
    print(' ')
