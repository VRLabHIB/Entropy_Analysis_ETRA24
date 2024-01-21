import numpy as np
import pandas as pd
import os
import glob

from utils.S101_create_graph_datasets import FullSessionDataset
from utils.S201_entropy_based_analysis_3AOIs import calculate_entropy_measures_3AOIs
from utils.helper import delete_files_in_directory
from utils.S201_entropy_based_analysis import calculate_entropy_measures

if __name__ == '__main__':
    project_path = os.path.abspath(os.getcwd())

    ### Big-Fish Dataset ####
    big_AOI_lst = ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S22', 'S23', 'S24', 'S27', 'S28', 'S32', 'S33',
                   'S34', 'S35', 'S36', 'S37', 'S38', 'S42', 'S43', 'S44', 'S47', 'S48', 'PresentationBoard', 'Teacher']
    # process big-fish dataset (create nodes and transitions)
    os.chdir("V:/Big-Fish_VR_Classroom/data/full_eye_tracking_data/")
    big_raw_lst = glob.glob("*.csv")
    big_save_path = project_path + "/data/big/"

    # get ID and condition information from file name
    big_name_df = pd.DataFrame({'name': big_raw_lst})
    big_name_df =big_name_df['name'].str.split('.', expand=True)[0].str.split('_', expand=True)[1]

    # clean nodes and transition folder
    # delete_files_in_directory(big_save_path)

    """
    print('Create Big-Fish Transition Datasets:')
    for i in range(len(big_raw_lst)):
        name = big_raw_lst[i]
        identifier = big_name_df.iloc[i][1:]
        condition = big_name_df.iloc[i][0]
        print('ID {}'.format(identifier))

        # create class that prepares the data for processing
        data = FullSessionDataset(name, identifier, condition, big_save_path, big_AOI_lst, 'big')

        # creates transition matrices and saves them into //data//nodes_and_transitions//
        data.create_one_node_dataset()
        data.create_one_transition_dataset()
        print(' ')
    """

    #### VirATeC Dataset ###
    vir_AOI_lst = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B', '5A', '5B', '6A', '6B', '7A', '7B', '8A', '8B',
                   '9A', '9B', 'PresentationBoard']

    # process viratec dataset (create nodes and transitions)
    os.chdir(r'V:\VirATeC\data\VirATeC_ADR\1_full_sessions')
    vir_raw_lst = glob.glob("*.csv")
    vir_save_path = project_path + "/data/vir/"

    # get ID and condition information from file name
    vir_name_df = pd.DataFrame({'name': vir_raw_lst})
    vir_name_df =vir_name_df['name'].str.split('.', expand=True)[0].str.split('D', expand=True)[1]

    # clean nodes and transition folder
    # delete_files_in_directory(vir_save_path)

    """
    print('Create VirATeC Transition Datasets:')
    for i in range(len(vir_raw_lst)):
        name = vir_raw_lst[i]
        identifier = vir_name_df.iloc[i]
        print('ID {}'.format(identifier))

        df_cond = pd.read_csv(project_path + '\\data\\questionnaire\\' + 'vir_condition.csv')
        condition = int(df_cond[df_cond['ID']==int(identifier)]['Expert'])

        # create class that prepares the data for processing
        data = FullSessionDataset(name, identifier, condition, vir_save_path, vir_AOI_lst,  'vir')

        # creates transition matrices and saves them into //data//nodes_and_transitions//
        data.create_one_node_dataset()
        data.create_one_transition_dataset()
    """

    ### Calculate Entropy ###
    # calculate_entropy_measures(project_path, big_AOI_lst, 'big')
    # calculate_entropy_measures(project_path, vir_AOI_lst, 'vir')
    calculate_entropy_measures_3AOIs(project_path, big_AOI_lst,'big')

    print('end')


