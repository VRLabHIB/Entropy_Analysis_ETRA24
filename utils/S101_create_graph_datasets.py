import os

import pandas as pd
import numpy as np

from utils.helper import calculate_time_diff, calculate_aoi_times, calculate_head_directionY_angle, \
    interpolate_AOI_objects, rename_AOI_names


def prepare_dataset(df, AOI_lst, datasets):
    if datasets == 'big':
        df = df.rename(columns={'real_time': 'Time', 'object': 'GazeTargetObject',
                                'distance_to_hit_object_gaze': 'RayDistanceGaze',
                                'left.pupil_diameter_mm': 'LeftPupilSize', 'right.pupil_diameter_mm': 'RightPupilSize'})

        df.insert(1, "TimeDiff", calculate_time_diff(df))
        df['HeadDirectionYAngle'] = calculate_head_directionY_angle(df, datasets)

        # convert Unreal to Unity coordinates
        # unreal X to Unity Z, Unreal Y to Unity X, Unreal Z to Unity Y
        df['GazeHitPointX'] = df['3D_hit_location_gaze.y']
        df['GazeHitPointY'] = df['3D_hit_location_gaze.z']
        df['GazeHitPointZ'] = df['3D_hit_location_gaze.x']

        # rename AOIs
        df['GazeTargetObject'] = rename_AOI_names(df, AOI_lst, datasets)

        # select subset of variables
        df = df[['Time', 'TimeDiff', 'GazeTargetObject', 'RightPupilSize', 'LeftPupilSize',
                 'HeadDirectionYAngle', 'RayDistanceGaze', 'GazeHitPointX', 'GazeHitPointY',
                 'GazeHitPointZ']]

        # cut end of experiment
        df = df[df['Time'] < 850]

        df = df[np.logical_or(df['RightPupilSize'] != -1, df['LeftPupilSize'] != -1)]

    if datasets=='vir':
        # cut end of experiment
        df = df[np.logical_and(df['Time'] >= 0, df['Time'] <= 600)]

        # rename AOIs
        df['GazeTargetObject'] = rename_AOI_names(df, AOI_lst, datasets)

        df['HeadDirectionYAngle'] = calculate_head_directionY_angle(df, datasets)

        # rename AOIs
        df['GazeTargetObject'] = rename_AOI_names(df, AOI_lst, datasets)

        df = df[['Time', 'TimeDiff', 'GazeTargetObject', 'RightPupilSize', 'LeftPupilSize',
                 'HeadDirectionYAngle', 'RayDistanceGaze', 'GazeHitPointX', 'GazeHitPointY',
                 'GazeHitPointZ']]

        df['RightPupilSize'] = df['RightPupilSize'].replace(np.nan, -1)
        df['LeftPupilSize'] = df['LeftPupilSize'].replace(np.nan, -1)
        df = df[np.logical_or(df['RightPupilSize'] != -1, df['LeftPupilSize'] != -1)]


    # clean dataset
    df['RightPupilSize'] = df['RightPupilSize'].replace(-1, np.nan)
    df['LeftPupilSize'] = df['LeftPupilSize'].replace(-1, np.nan)
    df['GazeTargetObject'] = interpolate_AOI_objects(df)
    df['GazeTargetObject'] = df['GazeTargetObject'].astype(str)
    df.insert(3, 'GazeTargetObject_Times', calculate_aoi_times(df, 'GazeTargetObject'))
    df = df[df['GazeTargetObject'] != 'none']
    return df


class FullSessionDataset:
    def __init__(self, name, identifier, condition, save_path, AOI_lst, datasets):
        if datasets == 'big':
            os.chdir("V:/Big-Fish_VR_Classroom/data/full_eye_tracking_data/")
        if datasets == 'vir':
            os.chdir(r'V:\VirATeC\data\VirATeC_ADR\1_full_sessions')

        self.df = pd.read_csv(name, low_memory=False)
        self.datasets = datasets
        self.identifier = identifier
        self.save_path = save_path
        self.condition = condition
        self.AOI_lst = AOI_lst

        # Clean and prepare the datasets
        self.df = prepare_dataset(self.df.copy(),  self.AOI_lst, datasets)

        self.df_lst = []

    def get_data(self):
        return self.df

    def get_matrices(self):
        return self.df_lst

    def get_ID(self):
        return self.identifier

    def create_one_node_dataset(self):

        # calculate pupil diameter baseline
        baselines_pupil_diameter = [np.nanmean(self.df['LeftPupilSize']), np.nanmean(self.df['RightPupilSize'])]

        # List of node attributes
        id_lst_node = list()
        condition_lst_node = list()
        duration_start_lst_node = list()
        source_lst_node = list()

        aoi_duration_lst_node = list()
        pupil_diameter_lst_node = list()
        distance_to_aoi_lst_node = list()


        # First source
        source = self.df['GazeTargetObject'].iloc[0]

        index = 0
        for i in range(1, len(self.df)):
            index += 1

            if source != self.df['GazeTargetObject'].iloc[i]:
                #### Select AOI interval ####
                dfs = self.df.iloc[i - index:i]

                id_lst_node.append(self.identifier)
                condition_lst_node.append(self.condition)
                duration_start_lst_node.append(dfs['Time'].iloc[0])
                source_lst_node.append(source)

                # AOI duration
                aoi_duration = dfs['GazeTargetObject_Times'].iloc[0]
                aoi_duration_lst_node.append(aoi_duration)

                # Average distance to gazes object
                avg_distance = np.nanmean(dfs['RayDistanceGaze'])
                distance_to_aoi_lst_node.append(avg_distance)

                # Pupil diameter
                df_p = pd.DataFrame(
                    {'LeftPupilBaselineCorrected': dfs['LeftPupilSize'] - baselines_pupil_diameter[0],
                     'RightPupilBaselineCorrected': dfs['RightPupilSize'] - baselines_pupil_diameter[1]})

                mean_pupil_diameter = np.nanmean(df_p, axis=1)
                pupil_diameter_lst_node.append(np.nanmean(mean_pupil_diameter))

                source = self.df['GazeTargetObject'].iloc[i]
                index -= index  # reset index

        # Last AOI in dataframe (not covert by the loop)
        dfs = self.df.iloc[i - index: i + 1]

        id_lst_node.append(self.identifier)
        condition_lst_node.append(self.condition)
        duration_start_lst_node.append(dfs['Time'].iloc[0])
        source_lst_node.append(source)

        # AOI duration
        aoi_duration = dfs['GazeTargetObject_Times'].iloc[0]
        aoi_duration_lst_node.append(aoi_duration)

        # Average distance to gazes object
        avg_distance = np.nanmean(dfs['RayDistanceGaze'])
        distance_to_aoi_lst_node.append(avg_distance)

        # Pupil diameter
        df_p = pd.DataFrame(
            {'LeftPupilBaselineCorrected': dfs['LeftPupilSize'] - baselines_pupil_diameter[0],
             'RightPupilBaselineCorrected': dfs['RightPupilSize'] - baselines_pupil_diameter[1]})

        mean_pupil_diameter = np.nanmean(df_p, axis=1)
        pupil_diameter_lst_node.append(np.nanmean(mean_pupil_diameter))

        #### Create both dataframes (node and egdes) ####
        df_node = pd.DataFrame({'ID': id_lst_node, 'Condition': condition_lst_node,
                                'duration_start': duration_start_lst_node,
                                'Target': source_lst_node,
                                'AOI_duration': aoi_duration_lst_node,
                                'pupil_diameter': pupil_diameter_lst_node,
                                'distance_to_aoi': distance_to_aoi_lst_node
                                })

        df_node.to_csv(self.save_path + str(self.datasets) + '_' + str(self.identifier) + '_node.csv',
                       index=False)

    def create_one_transition_dataset(self):
        # List of transition attributes
        id_lst_edge = list()
        condition_lst_edge = list()

        source_lst_edge = list()
        target_lst_edge = list()
        trans_start_lst_edge = list()
        trans_dur_lst_edge = list()
        weight_lst_edge = list()

        head_amplitude_lst_edge = list()
        trans_amplitude_lst = list()
        transition_velocity_lst = list()

        # First source
        source = self.df['GazeTargetObject'].iloc[0]

        index = 0
        for i in range(1, len(self.df)):
            index += 1

            if source != self.df['GazeTargetObject'].iloc[i]:
                trans_dur = self.df['Time'].iloc[i] - self.df['Time'].iloc[i - 1]
                if trans_dur < 1000:
                    trans_start_lst_edge.append(self.df['Time'].iloc[i - 1])
                    trans_dur_lst_edge.append(trans_dur)
                    source_lst_edge.append(source)
                    target_lst_edge.append(self.df['GazeTargetObject'].iloc[i])
                    weight_lst_edge.append(1)
                    id_lst_edge.append(self.identifier)
                    condition_lst_edge.append(self.condition)
                    head_amplitude_lst_edge.append(np.abs(self.df['HeadDirectionYAngle'].iloc[i]
                                                          - self.df['HeadDirectionYAngle'].iloc[i - 1]))

                    source_hitpoint = self.df[['GazeHitPointX', 'GazeHitPointY', 'GazeHitPointZ']].iloc[
                        i - 1].to_numpy()
                    target_hitpoint = self.df[['GazeHitPointX', 'GazeHitPointY', 'GazeHitPointZ']].iloc[i].to_numpy()
                    trans_amplitude = np.sum((target_hitpoint - source_hitpoint) ** 2)
                    trans_amplitude = np.sqrt(trans_amplitude)
                    trans_amplitude_lst.append(trans_amplitude)

                    transition_velocity_lst.append(trans_amplitude / trans_dur)

                    source = self.df['GazeTargetObject'].iloc[i]

        #### Create egdes dataframes####
        df_trans = pd.DataFrame({'ID': id_lst_edge, 'Condition': condition_lst_edge,
                                 'start_transition': trans_start_lst_edge,
                                 'Source': source_lst_edge, 'Target': target_lst_edge, 'Weight': weight_lst_edge,
                                 'trans_duration': trans_dur_lst_edge,
                                 'head_rotation_amplitude': head_amplitude_lst_edge,
                                 'trans_amplitude': trans_amplitude_lst,
                                 'trans_velocity': transition_velocity_lst,
                                 })

        df_trans.to_csv(self.save_path + str(self.datasets) + '_' + str(self.identifier) + '_trans.csv',
                        index=False)
