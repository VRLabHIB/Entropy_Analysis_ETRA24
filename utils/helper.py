# helper functions
import os

import numpy as np
from numpy.linalg import norm


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


def calculate_time_diff(df):
    """

    Parameters
    ----------
    df : pandas dataframe
        raw, cleaned dataframe.
    time : string
        variable that stores the time changes of an experiment [sec].

    Returns
    -------
    time_diff : list
        conveys time difference between following steps of an experiment.

    """

    # create list with start time of the experiment
    time_diff = np.array([0])

    # create two vectors
    t0 = np.array(df['Time'][:-1].tolist())
    t1 = np.array(df['Time'][1:].tolist())

    # vectors subtraction
    diff = np.subtract(t1, t0)

    time_diff = np.append(time_diff, diff)

    return time_diff


def calculate_aoi_times(df, var):
    time_lst = list()

    i = 0
    start_ooi = 'start'
    start_count = 0
    start_time = 0
    while i < len(df):
        ooi = df[var].iloc[i]

        if ooi != start_ooi:
            length = i - start_count
            duration = df['Time'].iloc[i] - start_time
            for j in range(length):
                time_lst.append(duration)

            start_ooi = df[var].iloc[i]
            start_count = i
            start_time = df['Time'].iloc[i]

        i += 1
    duration = df['Time'].iloc[len(df) - 1] - start_time
    j = 1
    for k in range(start_count, len(df)):
        time_lst.append(duration)

        j += 1

    return time_lst


def calculate_head_directionY_angle(df, datasets):
    # The head direction Y angle (angular difference from a straight forward head position)
    # Unity: Left-handed Y-Up coordinate system (x=to the right, y=up/down, z=forward/backward)
    # Unreal: left-handed Z-Up coordinate system (x=forward, y=to the right, z=up), clockwise rotation
    #         forward is roll=0, pitch=0, yaw=-90 (participants are faced towards the screen)
    #         yaw=left/right, pitch=up/down
    if datasets == 'big':
        angle = df['hmd.orientation.yaw'] - 90

    if datasets == 'vir':
        z = np.array([np.zeros(len(df)), np.ones(len(df))]).T
        v = df[['HeadDirectionX', 'HeadDirectionZ']].to_numpy()

        dot = np.einsum('ij,ij->i', z, v)
        no = norm(v, axis=1)
        sign = np.array(np.sign(np.cross(z, v, axis=1))) * (-1)
        angle = np.degrees(np.arccos(dot / no)) * sign
        angle = angle + 90

    return angle

def rename_AOI_names(df, AOI_lst, datasets):
    # different peer-groups acoording to their hand raising behaviour
    replace = {'S11_C': 'S11', 'S11_R': 'S11', 'S12_C': 'S12', 'S12_R': 'S12',
               'S13_C': 'S13', 'S13_R': 'S13', 'S14_C': 'S14', 'S14_R': 'S14',
               'S15_C': 'S15', 'S15_R': 'S15', 'S16_C': 'S16', 'S16_R': 'S16',
               'S17_C': 'S17', 'S17_R': 'S17', 'S22_C': 'S22', 'S22_R': 'S22',
               'S23_C': 'S23', 'S23_R': 'S23', 'S24_C': 'S24', 'S24_R': 'S24',
               'S27_C': 'S27', 'S27_R': 'S27', 'S28_C': 'S28', 'S28_R': 'S28',
               'S32_C': 'S32', 'S32_R': 'S32', 'S33_C': 'S33', 'S33_R': 'S33',
               'S34_C': 'S34', 'S34_R': 'S34', 'S35_C': 'S35', 'S35_R': 'S35',
               'S36_C': 'S36', 'S36_R': 'S36', 'S37_C': 'S34', 'S37_R': 'S37',
               'S38_C': 'S38', 'S38_R': 'S38', 'S42_C': 'S42', 'S42_R': 'S42',
               'S43_C': 'S43', 'S43_R': 'S43', 'S44_C': 'S44', 'S44_R': 'S44',
               'S47_C': 'S47', 'S47_R': 'S47', 'S48_C': 'S28', 'S48_R': 'S48',
               'CartoonTeacher': 'Teacher', 'RealTeacher_2': 'Teacher',
               'Screen_95': 'PresentationBoard'}

    if datasets=='big':
        df['GazeTargetObject'] = df['GazeTargetObject'].replace(replace)

    df['GazeTargetObject'] = np.where(df['GazeTargetObject'].isin(AOI_lst), df['GazeTargetObject'], 'none')
    return df['GazeTargetObject'].to_list()


def interpolate_AOI_objects(df):
    """
    :param target: GazeTargetObject
    :return: interpolated variable
    """
    object_var = 'GazeTargetObject'

    object_vector = df[object_var].to_numpy()

    # Remove single occurring aois
    if object_vector[0] != object_vector[1]:
        object_vector[0] = 'none'

    if object_vector[len(object_vector) - 1] != object_vector[len(object_vector) - 2]:
        object_vector[len(object_vector) - 1] = 'none'

    for row in range(1, len(df) - 1):
        aoi = object_vector[row]
        if aoi != 'none':
            if np.logical_and(aoi != object_vector[row - 1], aoi != object_vector[row + 1]):
                if object_vector[row - 1] == object_vector[row + 1]:
                    object_vector[row] = object_vector[row - 1]
                if object_vector[row - 1] != object_vector[row + 1]:
                    object_vector[row] = 'none'

    # Interpolate consecutive aois with max break of 2 nones
    # Find first aoi
    row = 0
    while object_vector[row] == 'none':
        row += 1
    # Now interpolate
    while row < len(object_vector):
        if object_vector[row] == 'none':
            last_aoi = object_vector[row - 1]

            length = 0
            while np.logical_and(row < len(object_vector) - 1, object_vector[row] == 'none'):
                row += 1
                length += 1

            next_aoi = object_vector[row]

            if np.logical_and(last_aoi == next_aoi, length < 3):
                start = row - length
                end = row
                object_vector[start:end] = last_aoi

        row += 1

    return object_vector
