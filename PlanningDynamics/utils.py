import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
import torch
import collections

def zscore(x, axis=0):
    """Standardise x to zero mean and unit variance along `axis`."""
    x_mean = np.mean(x, axis=axis, keepdims=True)
    x_std = np.std(x, axis=axis, keepdims=True)
    return (x - x_mean) / x_std

def confidence_interval(data, confidence=0.95, axis=0):
    """Calculate the confidence interval based on standard deviation."""
    mean = np.mean(data, axis)
    std_dev = np.std(data, axis)
    z_score = 1.96  # for 95% confidence
    margin_of_error = z_score * (std_dev)
    return mean - margin_of_error, mean + margin_of_error

def kl_divergence(p, q):
    """
    Calculates the KL divergence between two discrete probability distributions.

    Args:
        p (np.array): The reference probability distribution.
        q (np.array): The target probability distribution.

    Returns:
        float: The KL divergence value.
    """
    p = p / np.sum(p)
    q = q / np.sum(q)
    kl = np.sum(np.where((p != 0) & (q != 0), p * np.log(p / q), 0))
    return kl

def distance_to_value(dist, min_value=1):
    """Convert graph distance to reward value, clipped to [min_value, 4]."""
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

def check_overlap_lists(range1, range2):
    """
    Compute pairwise overlap lengths between two lists of (start, end) ranges.

    Returns an (len(range1), len(range2)) matrix of overlap durations.
    """
    overlaps = np.zeros([len(range1), len(range2)])
    for i, r1 in enumerate(range1):
        for j, r2 in enumerate(range2):
            start1, end1 = np.array(r1).T
            start2, end2 = np.array(r2).T
            overlap = np.maximum(0, np.minimum(end1, end2) - np.maximum(start1, start2))
            overlaps[i, j] = overlap
    return overlaps

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
    """Return a dict {subject: [sorted file paths]} for all NWB data files."""
    get_dir_files = lambda directory: sorted([str(f.resolve()) for f in Path(directory).iterdir() if f.is_file()])
    filename_dict = {}
    for sbj in ["bart", "london"]:
        dir = "data/" + sbj + "/"
        filename_dict[sbj] = get_dir_files(dir)
    return filename_dict

def get_ofc_fnames():
    """Return NWB file paths for OFC sessions, keyed by subject."""
    fnames = get_filenames()
    return fnames

def get_hpc_fnames():
    """Return NWB file paths for HPC sessions, keyed by subject."""
    fnames = get_filenames()
    del fnames["bart"][1]
    return fnames

def get_roi_fnames(region):
    """Return NWB file paths for the given region ('ofc' or 'hpc'), keyed by subject."""
    if region.lower() == "ofc":
        return get_ofc_fnames()
    elif region.lower() == "hpc":
        return get_hpc_fnames()
    else:
        raise ValueError("Invalid region specified. Choose 'ofc' or 'hpc'.")

def iterate_subjects(fnames, func):
    """Apply `func` to each file in `fnames` and collect results per subject."""
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
    """Extract key trial timestamps (start, first choice, reward onset) from a trial spike object."""
    evs, ets, eye_times = _d.evs, _d.ets, _d.eye_times
    t = eye_times - eye_times[0]
    
    ml_start_time = ets[np.where(evs == 9)[0][0]]
    
    ets = ets - ets[0]
    trial_start = ets[np.where(evs == 30)[0][0]]
    first_choice = ets[np.where(evs == 82)[0][0]] - 300
    reward_on = ets[np.where(evs == 71)[0][0]]
    return dict(t=t, start=trial_start, first_choice=first_choice, reward=reward_on, ml_start_time=ml_start_time)

def get_plan_data(data):
    """Extract planning-period spike data using a 100 ms minimum fixation duration and 0.2 active probability threshold."""
    plan_data = data.get_plan_spikes(type="mean", min_duration=100, active_prob_threshold=0.2)
    return plan_data

def find_consecutive_sequences(trace):
    """
    Find contiguous runs of non-zero values in a 1-D array.

    Returns an (n_sequences, 2) array of (start, end) indices, or None if all zeros.
    """
    if np.all(trace == 0):
        return None
    nonzero_indices = np.nonzero(trace)[0]
    diff = np.diff(nonzero_indices)
    split_indices = np.where(diff > 1)[0]
    sequences = np.split(nonzero_indices, split_indices + 1)
    sequences = [(seq[0], seq[-1]) for seq in sequences]
    
    return np.array(sequences)

def action_finish_times(codes):
    """
    parses reaction times from behavioral codes
    :param codes: dictionary of eventcodes (from get_bhvcodes)
    :return: (np.array) array of reaction times
    """
    num = codes["numbers"]
    times = codes["times"]
    t_finish_view = times[num == 82]
    t_finish_action = times[num == 92]
    t_finish_action_all = np.insert(t_finish_action, 0, t_finish_view[0])
    return t_finish_action_all

def finish_action_prev(codes):
    """
    Return the finish time of the preceding action for each step in a trial.

    The first element is the grid-on time; subsequent elements are the finish
    times of each action step, shifted so each entry aligns with the step that
    follows it.
    """
    num = codes["numbers"]
    times = codes["times"]
    t_grid_on = times[num == 30]
    t_finish_view = times[num == 82]
    t_finish_action = times[num == 92]
    t_finish_action_all = np.insert(t_finish_action, 0, t_finish_view[0])
    t_finish_action_prev = np.insert(t_finish_action_all[:-1], 0, t_grid_on)
    return t_finish_action_prev

def append_rts_and_planning(data):
    """
    Compute reaction times and planning metrics for all correct trials and
    return an augmented choice DataFrame.

    For each choice step, computes:
      - rt              : time from previous action offset to fixation onset
      - prev_plan       : number of prior fixations to the chosen node in this trial
      - any_plan        : 1 if any planning fixation occurred before this step
      - has_plan        : number of planning fixations concurrent with this step
      - value           : reward value derived from graph distance

    Planning fixations are defined as fixations with duration >= 50 ms,
    active_prob <= 0.2, and node_on != target.

    Parameters
    ----------
    data : nwbWrapper

    Returns
    -------
    choice_df : pd.DataFrame
    """
    trials = data.trial_df.query("trialerror == 0").trial.values
    prev_plan = []
    prev_action_offsets = []
    any_plan = []
    has_plan = []
    for trial in tqdm(trials):
        choices = data.choice_df.query("trial == @trial").copy()
        codes = dict(times=data.trial_spikes[trial].ets, numbers = data.trial_spikes[trial].evs)
        prev_action_offsets.append(finish_action_prev(codes))
        prev_plan_in_trial = np.zeros(choices.shape[0])
        any_plan_in_trial = np.zeros(choices.shape[0])
        has_plan_trial = np.zeros(choices.shape[0])
        plans = data.fixation_df.query("(duration >= 50) & (active_prob <= 0.2) & (node_on != target) & (trial == @trial)").copy()
        for i, choice in choices.reset_index().iterrows():
            has_plan_trial[i] = plans.query("step_on == %i" % choice.step).shape[0]
            prev_fix = plans.query("(fix_node == %i) & (step_off <= %i)" % (choice.node, choice.step)).loc[:, ("fix_node")]
            any_prev_fix = plans.query("(step_off <= %i)" % (choice.step)).loc[:, ("fix_node")]
            if prev_fix.shape[0] > 0:
                prev_plan_in_trial[i] = prev_fix.shape[0]
            if any_prev_fix.shape[0] > 0:
                any_plan_in_trial[i] = 1
        prev_plan.append(prev_plan_in_trial)
        any_plan.append(any_plan_in_trial)
        has_plan.append(has_plan_trial)
    has_plan = np.hstack(has_plan).flatten()    
    prev_plan = np.hstack(prev_plan).flatten()
    
    prev_action_offsets = np.hstack(prev_action_offsets).flatten()
    any_plan = np.hstack(any_plan).flatten()
    choice_df = data.choice_df.query("trialerror == 0").copy()
    choice_df["prev_plan"] = prev_plan
    choice_df["prev_action_offset"] = prev_action_offsets
    choice_df["rt"] = choice_df["fix_on"] - choice_df["prev_action_offset"]
    choice_df["has_plan"] = has_plan
    
    value = distance_to_value(choice_df.graph_distance.values)
    choice_df["value"] = value
    choice_df["any_plan"] = any_plan
    
    return choice_df