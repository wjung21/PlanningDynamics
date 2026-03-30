"""
Bootstrap Posterior Decoding for Neural Data

This module implements bootstrap-based decoding to estimate posterior probabilities
from neural activity during navigation. It builds multiple decoders trained on
bootstrapped samples to provide uncertainty estimates.

Main use cases:
1. Decode cognitive state (planning vs choice execution)
2. Decode choice value from neural activity during decisions
3. Decode plan value from neural activity during fixations

The bootstrap approach provides confidence intervals on decoded variables by training
an ensemble of classifiers on resampled data.
"""

import os
from PlanningDynamics import utils
import numpy as np
from PlanningDynamics.dataClass import nwbWrapper
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from copy import deepcopy

def make_classifier(clf=LinearDiscriminantAnalysis()):
    """
    Create a classification pipeline with z-scoring preprocessing.

    The pipeline standardizes neural firing rates before classification to ensure
    all neurons contribute equally regardless of baseline firing rate differences.

    Parameters
    ----------
    clf : sklearn classifier, optional
        The classifier to use (default: LinearDiscriminantAnalysis)

    Returns
    -------
    Pipeline
        A scikit-learn pipeline with StandardScaler followed by the classifier
    """
    return Pipeline([("zscore", StandardScaler()), ("clf", clf)])

def data_for_state_model(data):
    """
    Prepares data for a plan-vs-choice decoder by extracting relevant features and labels.
    
    Parameters
    ----------
    data : nwbWrapper
        Data object containing spike data and trial information.

    Returns
    -------
    tuple
        A tuple containing:
        - X (ndarray): Feature matrix (n_samples, n_neurons) combining plan and choice spikes
        - y (ndarray): Labels (n_samples, 1) where 1=planning, 0=choice
        - trial (ndarray): Trial IDs (n_samples,) for cross-validation
    """
    # Get planning data (fixations during deliberation)
    # - min_duration: only fixations lasting at least 100ms
    # - active_prob_threshold: exclude fixations overlapping with movement
    # - max_step_on: only early trial fixations (during planning phase)
    data_plan = data.get_plan_spikes(type="mean", min_duration=100, active_prob_threshold=0.1, max_step_on=10)

    # Exclude fixations that occurred after reaching the target
    plan_idx = (data_plan["df"]).node_off != (data_plan["df"]).target
    X_plan = data_plan["spikes"][plan_idx, :]
    plan_trial_id = data_plan["df"].trial.values[plan_idx]

    # Get choice data (neural activity during active movements)
    # Use second half of choice window (500ms onward) when movement is established
    X_choice = data.choice_spikes[:, 500:, :].mean(axis=1)
    choice_trial_id = data.choice_df.trial.values

    # Combine planning and choice data
    X = np.vstack([X_plan, X_choice])
    y = np.vstack([np.ones([X_plan.shape[0], 1]), np.zeros([X_choice.shape[0], 1])])
    trial = np.hstack([plan_trial_id, choice_trial_id])
    return X, y, trial

def data_for_choice_model(data):
    """
    Prepares data for a choice-value decoder.
    
    Parameters
    ----------
    data : nwbWrapper
        Data object containing choice_df and choice_spikes.

    Returns
    -------
    tuple
        A tuple containing:
        - X (ndarray): Neural firing rates (n_choices, n_neurons)
        - y (ndarray): Choice value labels (n_choices,) ranging from 1-4
        - trials (ndarray): Trial IDs (n_choices,) for cross-validation
    """
    # Convert graph distances to value categories (inverse relationship)
    y = data.choice_df.graph_distance.values
    y[y >= 3] = 3  # Bin distant locations together
    y = 4 - y      # Invert: smaller distance = higher value

    # Extract neural activity during choice execution (second half of window)
    X = data.choice_spikes[:, 500:, :].mean(axis=1)
    return X, y, data.choice_df.trial.values

def data_for_plan_model(data):
    """
    Prepares data for the plan-value decoder.

    Parameters
    ----------
    data : nwbWrapper
        Data object containing methods to retrieve plan spikes and associated data.

    Returns
    -------
    tuple
        A tuple containing:
        - X (ndarray): Neural firing rates during fixations (n_fixations, n_neurons)
        - y (ndarray): Value being planned (n_fixations,) ranging from 1-4
        - trials (ndarray): Trial IDs (n_fixations,) for cross-validation

    Notes
    -----
    Only includes fixations:
    - Lasting at least 100ms (deliberative, not saccadic)
    - Before reaching target (node ≠ target)
    """
    # Get fixation data during planning periods
    fix_data = data.get_plan_spikes(type="mean", min_duration=100, active_prob_threshold=0.2)

    # Exclude fixations after reaching the target
    idx = fix_data["df"]["node_on"] != fix_data["df"]["target"]
    X = fix_data["spikes"][idx, :]
    y = fix_data["df"].graph_distance.values[idx]

    # Convert graph distances to value categories (same encoding as choice model)
    y[y >= 3] = 3
    y = 4 - y

    return X, y, fix_data["df"].trial.values[idx]

def batch_trials(a, subset_length):
    """
    Partition trials into minibatches for cross-validation.

    Parameters
    ----------
    a : numpy.ndarray
        Array of trial IDs to partition
    subset_length : int
        Number of trials per minibatch (typically 10)

    Returns
    -------
    list
        List of numpy arrays, each containing trial IDs for one minibatch
    """
    a = np.random.permutation(a)  # Shuffle to avoid temporal biases
    return [a[i:i + subset_length] for i in range(0, len(a), subset_length)]

def train_idx_for_minibatch(X_by_trial, minibatch):
    """
    Create boolean mask for training data excluding held-out minibatch. This implements cross-validation by identifying which samples should be used
    for training by exculding all trials in the current minibatch).

    Parameters
    ----------
    X_by_trial : numpy.ndarray
        Array of trial IDs for each sample (n_samples,)
    minibatch : numpy.ndarray
        Array of trial IDs to hold out (n_trials_in_batch,)

    Returns
    -------
    numpy.ndarray
        Boolean array (n_samples,) where True = use for training, False = hold out

    Notes
    -----
    Uses broadcasting to check if each sample's trial ID matches any trial in the minibatch.
    """
    # Check if each sample belongs to any trial in the minibatch
    # Returns True for samples NOT in the minibatch (to be used for training)
    return (X_by_trial.reshape(-1, 1) == minibatch.reshape(1, -1)).sum(axis=1) == 0

def bootstrap_distribution(X, y, n_bootstraps=1000, n_samples_per_class=250):
    """
    Generate balanced bootstrap samples with replacement.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix (n_samples, n_features)
    y : numpy.ndarray
        Target labels (n_samples,)
    n_bootstraps : int, optional
        Number of bootstrap replicates (default: 1000)
    n_samples_per_class : int, optional
        Samples per class in each bootstrap (default: 250)

    Returns
    -------
    tuple
        - X_boot (ndarray): Bootstrapped features (n_bootstraps, n_samples_per_class * n_classes, n_features)
        - y_boot (ndarray): Bootstrapped labels (n_samples_per_class * n_classes,)

    """
    classes = np.unique(y)
    n_classes = len(classes)

    # Initialize arrays for bootstrapped data
    X_boot = np.zeros([n_bootstraps, n_samples_per_class*n_classes, X.shape[1]])
    y_boot = np.zeros([n_samples_per_class*n_classes])

    # Resample each class independently to maintain balance
    for i in range(n_classes):
        # Indices for this class in the output array
        idx = np.arange(n_samples_per_class*(i), n_samples_per_class * (i+1))

        # Get all samples from this class
        X_i = X[(y == classes[i]).flatten()]

        # Resample with replacement: draw n_bootstraps * n_samples_per_class total samples
        # then reshape to (n_bootstraps, n_samples_per_class, n_features)
        X_boot[:, idx, :] = resample(X_i, n_samples=n_bootstraps*n_samples_per_class, replace=True).reshape(n_bootstraps, n_samples_per_class, -1)
        y_boot[idx] = classes[i]

    return X_boot, y_boot

def bootstrap_posteriors(data, Xy_func, n_bootstraps, n_samples_per_class):
    """
    Generate time-varying posterior probabilities with bootstrapped confidence intervals. Combines cross-validation and bootstrapping.

    1. Divide trials into minibatches for cross-validation
    2. For each minibatch:
       a. Train n_bootstraps decoders on resampled data from other trials
       b. Apply each decoder to neural activity across time in held-out trials
       c. Store posterior probabilities from each bootstrap

    Parameters
    ----------
    data : nwbWrapper
        Data object containing trial spikes and behavioral information
    Xy_func : callable
        Function to prepare features and labels (e.g., data_for_choice_model)
    n_bootstraps : int
        Number of bootstrap replicates for uncertainty estimation
    n_samples_per_class : int
        Samples per class in each bootstrap training set

    Returns
    -------
    dict
        Dictionary with two keys:
        - posteriors: dict mapping trial_id -> (n_bootstraps, n_timepoints, n_classes) array
        - timesteps: dict mapping trial_id -> (n_timepoints,) array of timestamps

    Notes
    -----
    The posterior at time t, bootstrap b, class c is P(class=c | neural_activity_t, model_b).
    Averaging across bootstraps gives mean posterior, variance gives confidence.
    """
    # Prepare training data
    X, y, X_by_trial = Xy_func(data)
    trials = data.trial_df.query("trialerror == 0").trial.values  # Only successful trials
    n_classes = len(np.unique(y))

    clf = LinearDiscriminantAnalysis()
    posteriors = {}
    timesteps = {}

    # Group trials into minibatches for cross-validation (10 trials per batch)
    minibatches = batch_trials(trials, 10)

    for minibatch in tqdm(minibatches):
        # Get training data: all trials EXCEPT those in current minibatch
        train_idx = train_idx_for_minibatch(X_by_trial, minibatch)
        step_size = 10  # Decode every 10ms

        # Create bootstrap training sets by resampling with replacement
        X_boot, y_boot = bootstrap_distribution(X[train_idx], y[train_idx], n_bootstraps=n_bootstraps, n_samples_per_class=n_samples_per_class)

        # Train one decoder per bootstrap replicate
        models = []
        for j in range(n_bootstraps):
            model = make_classifier(clf=clf)
            model.fit(X_boot[j, ...], y_boot)
            models.append(deepcopy(model))  # Store fitted model

        # Apply all models to each held-out trial
        for trial in minibatch:
            # Create time axis (decode every step_size ms)
            t = np.arange(0, data.trial_spikes[trial].neural.shape[0], step_size).astype(np.int16)

            # Storage for posteriors from all bootstraps
            trial_posteriors = np.zeros([n_bootstraps, len(t), n_classes])

            # Extract neural activity in sliding windows (150ms total: 75ms before, 75ms after)
            X_trial = utils.matrix_square_window(data.trial_spikes[trial].neural, [75, 75], step_size)

            # Get posterior predictions from each bootstrap model
            for j in range(n_bootstraps):
                trial_posteriors[j, ...] = models[j].predict_proba(X_trial)

            # Store results (use float16 to save memory)
            posteriors[trial] = trial_posteriors.astype(np.float16)
            timesteps[trial] = t

    to_save = {"posteriors": posteriors, "timesteps": timesteps}
    return to_save

def save_posteriors(to_save, fname, var_name):
    """
    Save bootstrap posterior results to disk. Creates organized file structure: /data/bootstrapped_posteriors/{subject}/{session}_{var}_posterior_boot.pkl

    Parameters
    ----------
    to_save : dict
        Dictionary containing 'posteriors' and 'timesteps' from bootstrap_posteriors()
    fname : str
        Original data filename (used to extract subject and session info)
    var_name : str
        Variable name ('state', 'choice', or 'plan')
    """
    # Extract subject and session info from filename
    name = fname.split("/")[-1].split(".")[0]  # e.g., "London_TeleWorld_4x4_101124_spikes"
    sbj_name = name.split("_")[0].lower()      # e.g., "london"

    # Create standardized output filename
    savename = "/data/bootstrapped_posteriors/" + sbj_name + "/" + "_".join(name.split("_") + [var_name, "posterior_boot"]) + ".pkl"

    # Save as pickle file
    with open(savename, 'wb') as file:
        pickle.dump(to_save, file)
        print(f"Saved {savename}")

def compute_posteriors(fname):
    """
    Compute and save bootstrap posteriors for all three decoders. It generates three sets of posteriors:
    1. State: planning vs choice
    2. Choice: value being chosen (1-4)
    3. Plan: value being planned (1-4)

    Parameters
    ----------
    fname : str
        Path to NWB data file

    Notes
    -----
    Training set sizes differ by decoder complexity:
    - State (binary): 1000 samples per class (planning/choice well-represented)
    - Choice (4-way): 250 samples per class (limited by data availability)
    - Plan (4-way): 250 samples per class (limited by fixation counts)
    """
    
    vars = ["state", "choice", "plan"]
    funcs = [data_for_state_model, data_for_choice_model, data_for_plan_model]
    samples_per_class = [1000, 250, 250]  # Different sample sizes based on data availability

    # Load data from OFC (primary region for value encoding)
    data = nwbWrapper(fname, region="OFC")

    # Compute posteriors for each variable
    for n_samples, var, func in zip(samples_per_class, vars, funcs):
        to_save = bootstrap_posteriors(data, func, n_bootstraps=1000, n_samples_per_class=n_samples)
        save_posteriors(to_save, fname, var)
        print(f"Finished {var} for {fname}")
        del to_save  # Free memory before next decoder
       
        
def load_posteriors(fname, vars_to_load = ["state", "choice", "plan"]):
    """
    Load previously computed bootstrap posteriors.

    Parameters
    ----------
    fname : str
        Path to original NWB data file (used to identify corresponding posterior files)

    Returns
    -------
    dict
        Dictionary with keys 'state', 'choice', 'plan', each containing:
        - posteriors: dict mapping trial_id -> (n_bootstraps, n_timepoints, n_classes) array
        - timesteps: dict mapping trial_id -> (n_timepoints,) array

    Example
    -------
    >>> posteriors = load_posteriors("data/London_TeleWorld_4x4_101124_spikes.nwb")
    >>> state_post = posteriors["state"]["posteriors"][trial_id]  # (1000, n_times, 2)
    >>> mean_state = state_post.mean(axis=0)  # Average across bootstraps
    """
    # Extract session identifier from filename
    name = fname.split("/")[-1].split(".")[0]
    sbj_name = name.split("_")[0].lower()
    posteriors_directory = "data/bootstrapped_posteriors/" + sbj_name + "/"

    # Find all posterior files for this session
    filenames = os.listdir(posteriors_directory)
    file_post = []
    for file in filenames:
        if (name in file) & ("posterior" in file):
            file_post.append(file)

    # Load each decoder's output
    posteriors = {}
    for post in file_post:
        for var in vars_to_load:
            if var in post:
                with open(posteriors_directory + post, 'rb') as file:
                    posteriors[var] = pickle.load(file)

    return posteriors

def find_states(trace, prob_threshold = 0.5, duration_threshold = 4, step_size = 5):
    n = trace.shape[0]
    t = np.arange(0, n, step_size)
    trace = trace[t]
    seq = utils.find_consecutive_sequences(trace > prob_threshold)
    if seq is None:
        return None
    idx_to_keep = np.where((seq[:, 1] - seq[:, 0]) >= duration_threshold)[0]
    return t[seq[idx_to_keep]]


def state_df_boot(trial, data, state_posteriors, plan_posteriors, choice_posteriors):
    """
    Detect planning states during choice and extract decoded values for a single trial. Planning states are detected using 95% CI lower bound > 0.5 threshold. States must overlap ≥75% with choice fixation to be included.

    Parameters
    ----------
    trial : int
        Trial number to analyze
    data : nwbWrapper
        Data object with behavioral and neural data
    state_posteriors : dict
        Output from state decoder (planning vs choice)
    plan_posteriors : dict
        Output from plan value decoder (values 1-4)
    choice_posteriors : dict
        Output from choice value decoder (values 1-4)

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per detected planning state, containing:
        - trial: trial number
        - step: which choice/fixation this overlaps with
        - plan_value: decoded value being planned (1-4)
        - choice_value: decoded value being chosen (1-4)
        - current_value: actual value at current location
        - duration: duration of planning state (ms)
        - state_on/off: onset/offset times (ms from trial start)
        - dist: graph distance to target
        - nodes: path taken
        - nsteps: total steps in trial
    """
    if 'value' not in data.choice_df.columns:
        data.choice_df["value"] = utils.distance_to_value(data.choice_df.graph_distance.values)

    # Time axis for decoding (step 10ms)
    timesteps = np.arange(0, data.trial_spikes[trial].neural.shape[0], 10)

    # Get choice times for this trial
    trial_steps = data.choice_df.query("trial == @trial")
    _d = data.trial_spikes[trial]
    ts = utils.trial_ts(_d)
    step_on = trial_steps["fix_on"].values - ts["ml_start_time"]
    step_off = trial_steps["fix_off"].values - ts["ml_start_time"]

    # Detect planning states using conservative threshold (95% CI lower bound)
    ci95 = utils.confidence_interval(state_posteriors[trial][:, :, 1], confidence=0.95, axis=0)
    threshold_trace = utils.movmean(ci95[0], 2).flatten()  # Smooth threshold
    states = find_states(threshold_trace, prob_threshold=0.5, duration_threshold=1, step_size=1)

    if states is None:
        return pd.DataFrame()  # No planning states detected

    # Decode values during each planning state
    plan_pred = plan_posteriors[trial].mean(axis=0)    # Average across bootstraps
    choice_pred = choice_posteriors[trial].mean(axis=0)
    state_plan_pred = np.zeros(len(states))
    state_choice_pred = np.zeros(len(states))

    # For each planning state, get the most likely value
    for i, state in enumerate(states):
        # Most common predicted value during this state
        state_plan_pred[i] = np.argmax(plan_pred[state[0]:state[1], :].mean(axis=0)) + 1
        state_choice_pred[i] = np.argmax(choice_pred[state[0]:state[1], :].mean(axis=0)) + 1

    # Convert state indices to timestamps
    states = timesteps[states]
    durations = (states[:, 1] - states[:, 0]).reshape(-1, 1)

    # Match planning states to behavioral fixations (must overlap ≥75%)
    state_overlap_with_fix = utils.check_overlap_lists(states, np.hstack([step_on.reshape(-1, 1), step_off.reshape(-1, 1)])) / durations.reshape(-1, 1)
    valid_state_id, fix_id = np.where(state_overlap_with_fix >= 0.75)

    # Create output DataFrame
    state_dict = dict(trial = trial,
                    step = fix_id,
                    plan_value = state_plan_pred[valid_state_id].astype(int),
                    choice_value = state_choice_pred[valid_state_id].astype(int),
                    current_value = trial_steps.iloc[fix_id, :].value.values,
                    duration = durations[valid_state_id, 0],
                    dist = trial_steps.iloc[fix_id, :].graph_distance.values,
                    ml_start_time = ts["ml_start_time"],
                    state_on = states[valid_state_id, 0],
                    state_off = states[valid_state_id, 1],
                    nodes = trial_steps.iloc[fix_id, :].nodes.values,
                    nsteps = len(trial_steps),
                    )
    state_df = pd.DataFrame(state_dict)
    return state_df
