import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
import os
import torch
import collections

def distance_to_value(dist, min_value=1):
    return np.clip(4 - dist, min_value, 4)

def check_overlap(range1, range2):
    """
    Check if two ranges overlap.

    Args:
        range1 (tuple): A tuple representing the first range (start, end).
        range2 (tuple): A tuple representing the second range (start, end).

    Returns:
        bool: True if the ranges overlap, False otherwise.
    """
    
    if range1[0] <= range2[0] <= range1[1] or range2[0] <= range1[0] <= range2[1]:
        return True
    return False

def matrix_square_window(spikes, window_size, step_size, zscore=False):
    """
    Computes the mean of spike data over a sliding window.

    Parameters:
    spikes (numpy.ndarray): A 2D array where rows represent time steps and columns represent neurons.
    window_size (int or list): Size of the window for averaging. If an integer is provided, it is used for both dimensions.
    step_size (int): The step size for moving the window.
    zscore (bool): If True, the output is z-scored. Default is False.

    Returns:
    numpy.ndarray: A 2D array where each row corresponds to the mean spike activity over the window at each step.
    """
    
    if type(window_size) == int:
        window_size = [window_size, window_size]
        
    n_timesteps, n_neurons = spikes.shape
    steps = np.arange(0, n_timesteps, step_size)
    Y = np.zeros([len(steps), n_neurons])
    for i, s in enumerate(steps):
        window_start = np.maximum(0, s - window_size[0])
        window_end = np.minimum(n_timesteps, s + window_size[1])
        if zscore:
            Y[i, :] = StandardScaler().fit_transform((spikes[window_start:window_end, :].mean(axis=0)))
        else:
            Y[i, :] = spikes[window_start:window_end, :].mean(axis=0)
    return Y

def get_filenames():
    current_directory = os.getcwd()
    get_dir_files = lambda directory: sorted([str(f.resolve()) for f in Path(directory).iterdir() if f.is_file()])
    filename_dict = {}
    for sbj in ["bart", "london"]:
        #dir = current_directory + "/data/" + sbj + "/"
        dir = "/media/eric/partition_1/PlanningDynamics/data/" + sbj + "/"
        filename_dict[sbj] = get_dir_files(dir)
    return filename_dict

def iterate_subjects(fnames, func):
    res = {"bart":[], "london":[]}
    for sbj in ["bart", "london"]:
        for i in tqdm(range(len(fnames[sbj])), desc=f"Processing {sbj}"):
            out = func(fnames[sbj][i])
            if out is not None:
                res[sbj].append(out)
    return res

def get_rts(codes):
    """
    parses reaction times from behavioral codes
    :param codes: dictionary of eventcodes (from get_bhvcodes)
    :return: (np.array) array of reaction times
    """
    num = codes["numbers"]
    times = codes["times"]
    t_grid_on = times[num == 30]
    t_finish_view = times[num == 82]
    t_finish_action = times[num == 92]
    t_action_start = t_finish_action - 300
    if len(t_finish_view) == 0:
        return None
    first_step_rt = t_finish_view[0] - 300 - t_grid_on
    if len(t_action_start) == len(t_finish_view)-1:
        action_steps_rt = t_action_start - t_finish_view[: -1]
    else:
        action_steps_rt = t_action_start - np.hstack([t_finish_view[: -1], np.nan])
    rts = np.vstack([first_step_rt, action_steps_rt.reshape(-1, 1)])
    return np.ravel(rts)

def movmean(A, w, gpu=False):
    """
    Calculates the moving mean of a matrix or vector horizontally.

    Args:
        A (np.array): The input matrix or vector. It must be <= 2 dimensions.
        w (int or list or ndarray): The window size for calculating the moving mean.
            If it is an integer, it calculates a centered window of size w around each element.
            If it is a 2-element list or ndarray, it calculates the window w1 elements before and w2 elements after.

    Returns:
        np.array: The moving mean of the input matrix or vector.
    """

    if A.ndim > 2:
        print("Error: input matrix cannot exceed 2 dimensions")
    elif A.ndim <= 1:
        A = A.reshape(1, -1)
    n = A.shape[1]
    if isinstance(w, (collections.abc.Sequence, np.ndarray)):
        params = [w[0], w[1]]
    else:
        params = [w, w]
        
    if gpu:
        A = torch.tensor(A, device="cuda").to(torch.float32)
        weight_matrix = torch.tril(torch.triu((torch.ones([n, n])), -params[1]), params[0]).cuda().to(dtype=torch.float32)
        return ((A @ weight_matrix)/weight_matrix.sum(axis=0)).cpu().numpy()
    else:
        weight_matrix = np.tril(np.triu((np.ones([n, n])), -params[1]), params[0])
        return np.dot(A, weight_matrix)/weight_matrix.sum(axis=0)
        
def trial_ts(_d):
    
    evs, ets, eye_times = _d.evs, _d.ets, _d.eye_times
    t = eye_times - eye_times[0]
    
    ml_start_time = ets[np.where(evs == 9)[0][0]]
    
    ets = ets - ets[0]
    trial_start = ets[np.where(evs == 30)[0][0]]
    first_choice = ets[np.where(evs == 82)[0][0]] - 300
    reward_on = ets[np.where(evs == 71)[0][0]]
    return dict(t=t, start=trial_start, first_choice=first_choice, reward=reward_on, ml_start_time=ml_start_time)

