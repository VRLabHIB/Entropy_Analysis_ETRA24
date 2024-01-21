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

    ### Calculate Entropy ###
    # calculate_entropy_measures(project_path, big_AOI_lst, 'big')
    # calculate_entropy_measures_3AOIs(project_path, big_AOI_lst,'big')

    print('end')


