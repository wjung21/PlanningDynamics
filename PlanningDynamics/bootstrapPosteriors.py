from PlanningDynamics import utils
import numpy as np
from PlanningDynamics.dataClass import nwbWrapper
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pickle

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from copy import deepcopy

def make_classifier(clf=LinearDiscriminantAnalysis()):
    return Pipeline([("zscore", StandardScaler()), ("clf", clf)])

def data_for_state_model(data):
    """
    Prepares data for a plan-vs-choice decoder by extracting relevant features and labels.

    Parameters
    ----------
    data : object
        An object containing methods and attributes to access spike data and trial information.

    Returns
    -------
    tuple
        A tuple containing:
        - X (ndarray): A 2D array of features combining plan and choice spikes.
        - y (ndarray): A 2D array of labels (1 for plan, 0 for choice).
        - trial (ndarray): A 1D array of trial identifiers corresponding to the features.
    """
    # get planning data
    data_plan = data.get_plan_spikes(type="mean", min_duration=100, active_prob_threshold=0.1, max_step_on=10)
    plan_idx = (data_plan["df"]).node_off != (data_plan["df"]).target
    X_plan = data_plan["spikes"][plan_idx, :]
    plan_trial_id = data_plan["df"].trial.values[plan_idx]
    
    # get choice data
    X_choice = data.choice_spikes[:, 500:, :].mean(axis=1)
    choice_trial_id = data.choice_df.trial.values

    # combine
    X = np.vstack([X_plan, X_choice])
    y = np.vstack([np.ones([X_plan.shape[0], 1]), np.zeros([X_choice.shape[0], 1])])
    trial = np.hstack([plan_trial_id, choice_trial_id])
    return X, y, trial

def data_for_choice_model(data):
    """
    Prepares data for a choice-value decoder by processing the input data.

    Parameters
    ----------
    data : object
        An object containing choice_df and choice_spikes attributes. choice_df should have a 'graph_distance' column and 'trial' values. 
        choice_spikes should be a 3D numpy array.

    Returns
    -------
    tuple
        A tuple containing:
        - X (ndarray): The mean of choice_spikes across the specified axis.
        - y (ndarray): The value of each choice, ranging from 1-4.
        - trials (ndarray): The trial values from choice_df.
    """
    # convert graph distances to values 
    y = data.choice_df.graph_distance.values
    y[y >= 3] = 3
    y = 4 - y
    
    # get spikes
    X = data.choice_spikes[:, 500:800, :].mean(axis=1)
    return X, y, data.choice_df.trial.values

def data_for_plan_model(data):
    """
    Prepares data for the plan-value decoder by extracting relevant features and labels.

    Parameters
    ----------
    data : object
        An object containing methods to retrieve plan spikes and associated data.

    Returns
    -------
    tuple
        A tuple containing:
        - X (ndarray): The feature matrix of spikes where the node is not the target.
        - y (ndarray): The value of each choice, ranging from 1-4.
        - trials (ndarray): The trial values corresponding to the selected spikes.
    """
    
    # get plan data
    fix_data = data.get_plan_spikes(type="mean", min_duration=100, active_prob_threshold=0.2, max_step_on=10)
    idx = fix_data["df"]["node_on"] != fix_data["df"]["target"]
    X = fix_data["spikes"][idx, :]
    y = fix_data["df"].graph_distance.values[idx]
    
    # convert graph distances to values 1-4
    y[y >= 3] = 3
    y = 4 - y
    
    return X, y, fix_data["df"].trial.values[idx]

def batch_trials(a, subset_length):
    """
    Partition data into subsets of equal length.

    Parameters
    ----------
    a : numpy.ndarray
        The array to be partitioned.
    subset_length : int
        The length of each subset.

    Returns
    -------
    list
        A list of numpy arrays, each containing a subset of the original array.
    """
    a = np.random.permutation(a)  # Shuffle the data
    return [a[i:i + subset_length] for i in range(0, len(a), subset_length)]

def train_idx_for_minibatch(X_by_trial, minibatch):
    """
    Determine indices of training data that are not part of the given minibatch.

    Parameters
    ----------
    X_by_trial : numpy.ndarray
        Array of training data organized by trial.
    minibatch : numpy.ndarray
        Array representing the minibatch to exclude.

    Returns
    -------
    numpy.ndarray
        Boolean array indicating which indices in X_by_trial are not included in the minibatch.
    """
    return (X_by_trial.reshape(-1, 1) == minibatch.reshape(1, -1)).sum(axis=1) == 0

def bootstrap_distribution(X, y, n_bootstraps=1000, n_samples_per_class=250):
    """
    Generate bootstrap samples for a given dataset.

    Parameters
    ----------
    X : numpy.ndarray
        The feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        The target vector of shape (n_samples,).
    n_bootstraps : int, optional
        The number of bootstrap samples to generate (default is 1000).
    n_samples_per_class : int, optional
        The number of samples to draw for each class (default is 250).
    random_state : int, optional
        Seed for the random number generator (default is 42).

    Returns
    -------
    tuple
        A tuple containing:
        - X_boot (numpy.ndarray): The bootstrapped feature matrix of shape 
            (n_bootstraps, n_samples_per_class * n_classes, n_features).
        - y_boot (numpy.ndarray): The bootstrapped target vector of shape 
            (n_samples_per_class * n_classes,).
    """
    classes = np.unique(y)
    n_classes = len(classes)
    # create empty arrays for bootstrapped data
    X_boot = np.zeros([n_bootstraps, n_samples_per_class*n_classes, X.shape[1]])
    y_boot = np.zeros([n_samples_per_class*n_classes])
    
    # resample data for each class
    for i in range(n_classes):
        idx = np.arange(n_samples_per_class*(i), n_samples_per_class * (i+1))
        X_i = X[(y == classes[i]).flatten()]
        X_boot[:, idx, :] = resample(X_i, n_samples=n_bootstraps*n_samples_per_class, replace=True).reshape(n_bootstraps, n_samples_per_class, -1)
        y_boot[idx] = classes[i]
    return X_boot, y_boot

def bootstrap_posteriors(data, Xy_func, n_bootstraps, n_samples_per_class):
    """
    Generate bootstrap posteriors for a given dataset.

    Parameters
    ----------
    data : object
        The data object containing trial spikes and other relevant information.
    Xy_func : callable
        A function that prepares the data and returns features and labels.
    n_bootstraps : int
        The number of bootstrap samples to generate.
    n_samples_per_class : int
        The number of samples to draw for each class.

    Returns
    -------
    dict
        A dictionary containing:
        - posteriors: A dictionary of trial posteriors.
        - timesteps: A dictionary of timesteps corresponding to each trial.
    """
    X, y, X_by_trial = Xy_func(data)
    trials = data.trial_df.query("trialerror == 0").trial.values
    n_classes = len(np.unique(y))
    
    clf = LinearDiscriminantAnalysis()
    posteriors = {}
    timesteps = {}
    
    # group trials into minibatches of 10
    minibatches = batch_trials(trials, 10)

    for minibatch in tqdm(minibatches):
        # get training data excluding the current minibatch
        train_idx = train_idx_for_minibatch(X_by_trial, minibatch)
        step_size=10
        
        # resample training data with replacement to create bootstrap samples
        X_boot, y_boot = bootstrap_distribution(X[train_idx], y[train_idx], n_bootstraps=n_bootstraps, n_samples_per_class=n_samples_per_class)
        
        # fit a model to each bootstrap sample
        models = []
        for j in range(n_bootstraps):
            model = make_classifier(clf=clf)
            model.fit(X_boot[j, ...], y_boot)
            models.append(deepcopy(model))
            
        # get posteriors for each trial in the minibatch
        for trial in minibatch: 
            t = np.arange(0, data.trial_spikes[trial].neural.shape[0], step_size).astype(np.int16)
            trial_posteriors = np.zeros([n_bootstraps, len(t), n_classes])    
            X_trial = utils.matrix_square_window(data.trial_spikes[trial].neural, [75, 75], step_size)
            for j in range(n_bootstraps):
                trial_posteriors[j, ...] = models[j].predict_proba(X_trial)
            posteriors[trial] = trial_posteriors.astype(np.float16)
            timesteps[trial] = t
        to_save = {"posteriors": posteriors, "timesteps": timesteps}
        return to_save

def save_posteriors(to_save, fname, var_name):
    name = fname.split("/")[-1].split(".")[0]
    
    savename = "/data/bootstrapped_posteriors/" + "_".join(name.split("_") + [var_name, "posterior_boot"]) + ".pkl"
    with open(savename, 'wb') as file:
        pickle.dump(to_save, file)
        print(f"Saved {savename}")

def compute_posteriors(fname):
    """
    Save bootstrapped posterior samples for specified variables.

    This function retrieves data from the specified file and generates
    bootstrapped posterior samples for the variables: state, choice, and plan.
    It uses predefined functions to obtain the necessary data for each variable
    and saves the results to the specified file.

    Parameters:
    fname : str
        The filename from which to retrieve data and to which results will be saved.

    Returns:
    None
    """
    vars = ["state", "choice", "plan"]
    funcs = [data_for_state_model, data_for_choice_model, data_for_plan_model]
    samples_per_class = [1000, 250, 250]
    data = nwbWrapper(fname, region="OFC")
    for n_samples, var, func in zip(samples_per_class, vars, funcs):
        to_save = bootstrap_posteriors(data, func, n_bootstraps=1000, n_samples_per_class=n_samples)
        save_posteriors(to_save, fname, var)
        print(f"Finished {var} for {fname}")
        del to_save
        