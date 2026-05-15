from PlanningDynamics import utils, graph
import numpy as np
import pynwb
from tqdm import tqdm

from collections import namedtuple
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # to ignore numpy DeprecationWarning


# namedtuple to hold data for each trial 
TrialData = namedtuple("TrialData", ["neural", "evs", "ets", "eyes", "eye_times", "trialerror"]) 
# neural: time x neurons, evs: event data, ets: event timestamps, eyes: eye data, eye_times: eye timestamps, trialerror: trial error code
 

def get_unitNames(nwbfile, region, drift, min_fr):
    unitNames = nwbfile.units.to_dataframe()
    if region.lower() == "both":
        idx = (unitNames.fr >= min_fr) & (unitNames.drift <= drift) & (unitNames.group == "good")
    else:
        idx = (unitNames.fr >= min_fr) & (unitNames.region == region) & (unitNames.drift <= drift) & (unitNames.group == "good")
    return unitNames[idx], idx
    
def get_trial_data(nwbfile, trials, unit_idx):
    """
    Retrieves trial data from a NWB (Neurodata Without Borders) file.

    Parameters:
    nwb (NWBFile): The NWB file containing the data.
    trials (list or str): A list of trial indices or "all" to select all trials.
    region (str): The region of interest (e.g., "both", "OFC", or "HPC").
    drift (float, optional): The maximum allowed drift value for units, which were manually defined from sorting. Default is 2.
    
    Returns:
    tuple: A dictionary containing trial data for each trial and a DataFrame of trial metadata.
    """
    out = {}
    trial_df = nwbfile.trials.to_dataframe()
    
    if trials == "all":
        trials = np.where(trial_df.trialerror <= 2)[0]
    for trial in trials:        
        neural = trial_df.iloc[trial].timeseries[0].data[:, unit_idx]
        evs = trial_df.iloc[trial].timeseries[1].data
        ets = trial_df.iloc[trial].timeseries[1].timestamps
        eyes = trial_df.iloc[trial].timeseries[2].data
        eye_times = trial_df.iloc[trial].timeseries[2].timestamps
        out[trial] = TrialData(neural, evs, ets, eyes, eye_times, trial_df.iloc[trial].trialerror)
    return out, trial_df.iloc[trials, :].drop(["timeseries"], axis=1)

def get_choice_data(nwbfile, unit_idx, epoch="action_on", query="(trialerror == 0)"):
    """
    Retrieves spike data and unit information from an NWB file.

    Args:
        filename (str): The path to the NWB file.
        region (str): The region of interest.
        window_size (int, optional): The size of the spike window. Defaults to 0.
        drift (int, optional): The maximum drift allowed. Defaults to 2.
        epoch (str, optional): The epoch to consider. Defaults to "action_on".
        min_fr (int, optional): The minimum firing rate. Defaults to 1.
        query (str, optional): The query to filter trials. Defaults to "(trialerror < 2)".

    Returns:
        dict: A dictionary containing the spike data, dataframe, and unit names.
    """
        # Allows subselection of trials based on a pandas query
    if query == "":
        choice_df = nwbfile.intervals[epoch].to_dataframe()#["timeseries"]
    else:
        choice_df = nwbfile.intervals[epoch].to_dataframe().query(query)#["timeseries"]
    
    # Index of neural timeseries in NWB file 
    neural_timeseries_index = 0
    
    # Get the window size from the first entry in the dataframe
    epoch_win_size = choice_df["window_size"].values[0]
    
    # Initialize an array to hold the spike data
    sample_index = choice_df.index
    choice_data = np.zeros([len(sample_index), epoch_win_size*2, sum(unit_idx)])
    
    # Loop through each sample and extract neural data
    # take the moving average of the neural data if window_size > 0
    for i, _sample in enumerate(sample_index):
        choice_data[i, ...] = choice_df["timeseries"][_sample][neural_timeseries_index].data[:, unit_idx]
    choice_df = graph.append_use_tele(choice_df)
    return choice_data, choice_df

def get_fixation_table(nwbfile): 
        fix_table = nwbfile.intervals["fixations"].to_dataframe()
        g = graph.grid_graph(4, 4, tele=[0, 15])
        fix_dist = fix_table.apply(lambda row: graph.optimal_steps(g, row.fix_node, row.target), axis=1)
        fix_table["graph_distance"] = fix_dist
        return fix_table.drop(["timeseries"], axis=1)

# improvements: Should only read NWB file once
class nwbWrapper:
    def __init__(self, fname, region, to_load="all", choice_query="(trialerror == 0)"):
        self.fname = fname
        self.region = region
        self.sbj = fname.split("/")[-1].split("_")[0].lower()
        self.session = fname.split("/")[-1].split("_")[-1]
        self.choice_query = choice_query
        if to_load not in ["all", "bhv"]:
            raise ValueError("to_load must be 'all' or 'trial_df'")
        
        if to_load == "bhv":
            self.trial_df, self.choice_df, self.fixation_df = self.load_bhv_data()
        
        elif to_load == "all":
            self.load_all_data()
            self.n_cells = self.unitNames.shape[0]
            
        

               
    def load_all_data(self):
        with pynwb.NWBHDF5IO(self.fname, mode='r') as io:
            nwbfile = io.read()  
            # get neuron info 
            self.unitNames, unit_idx = get_unitNames(nwbfile, self.region, drift=2, min_fr=1)
            
            # get data aligned to the onset of each trial
            self.trial_spikes, self.trial_df = get_trial_data(nwbfile, "all", unit_idx)
            
            # get data aligned to the onset of each choice
            self.choice_spikes, self.choice_df = get_choice_data(nwbfile, unit_idx, epoch="action_on", query=self.choice_query)
            
            # load data segmented to fixations
            self.fixation_df = get_fixation_table(nwbfile)
        
        self.get_choice_fix_times() # calculate the onset and offset of choices based off eye movements, not just by eventcodes 



    def load_trial_df(self):
        with pynwb.NWBHDF5IO(self.fname, mode='r') as io:
            nwbfile = io.read()  
            trial_df = nwbfile.trials.to_dataframe().query("(trialerror <= 2)").copy()
        return trial_df.drop(["timeseries"], axis=1)
            
    def load_bhv_data(self):
        with pynwb.NWBHDF5IO(self.fname, mode='r') as io:
            nwbfile = io.read()  
            trial_df = nwbfile.trials.to_dataframe().query("(trialerror <= 2)").copy()
            choice_df = nwbfile.intervals["action_on"].to_dataframe().query("trialerror <= 2")
            choice_df = graph.append_use_tele(choice_df)
            fixation_df = get_fixation_table(nwbfile)
        
        return trial_df.drop(["timeseries"], axis=1), choice_df.drop(["timeseries"], axis=1), fixation_df        

    def get_choice_fix_times(self):
        """ Matches fixations identified using eventcodes to fixations detected using eyetracking. Resolves task fixations where multiple eye-tracking fixations are detected.
        """
        self.choice_df
        fix_on = np.zeros(self.choice_df.shape[0]) - 100
        fix_off = np.zeros(self.choice_df.shape[0]) - 100
        for trial in self.fixation_df.trial.unique():

            fixes = self.fixation_df[self.fixation_df.trial == trial]
            choices = self.choice_df[self.choice_df.trial == trial]
            
            for _, (_, step) in enumerate(choices.iterrows()):
                step_dur = (step.t_on, step.t_on+300)
                fixes_for_step = np.where(fixes.apply(lambda x: utils.check_overlap(step_dur, (x.t_on, x.t_off)), axis=1).values)[0]
                step_idx = np.where((self.choice_df.trial == trial) & (self.choice_df.step == step.step))[0][0]

                if len(fixes_for_step) == 1:
                    row = fixes.iloc[fixes_for_step]
                    
                    fix_on[step_idx] = int(row.t_on.values)
                    fix_off[step_idx] = int(row.t_off.values)
                    
                elif len(fixes_for_step) == 2:
                    n_unique = lambda x: len(np.unique(x))
                    
                    if n_unique(fixes.iloc[fixes_for_step].fix_node.values) == 1:
                        # Same node
                        fix_on[step_idx] = int(fixes.iloc[fixes_for_step].t_on.values[0])
                        fix_off[step_idx] = int(fixes.iloc[fixes_for_step].t_off.values[1])
                    else:
                        # Different nodes
                        node = step.node
                        fix_to_keep = fixes_for_step[np.where(fixes.iloc[fixes_for_step].fix_node.values == node)[0]]
                        fix_on[step_idx] = int(fixes.iloc[fix_to_keep].t_on.values)
                        fix_off[step_idx] = int(fixes.iloc[fix_to_keep].t_off.values)
                else:
                    fix_on[step_idx] = int(step.t_on)
                    fix_off[step_idx] = int(step.t_on) + 300
                    
        self.choice_df["fix_on"] = fix_on.astype(int)
        self.choice_df["fix_off"] = fix_off.astype(int)
        return self.choice_df


    def get_plan_spikes(self, type="mean", min_duration = 100, max_duration = 1000, max_step_on = 10, active_prob_threshold = 0.2, window_size=500):
        assert type in ["mean", "psth"] 
        
        fix_values = []
        fix_spikes = []
        fix_trial_id = []
        n_cells = self.trial_spikes[0].neural.shape[1]

        trials = np.array(list(self.trial_spikes.keys()))
        fix_df = self.fixation_df.copy()
        fix_df = fix_df[(fix_df.step_on <= max_step_on) & # exclude fixations after a certain step
                        (fix_df.duration >= min_duration) & (fix_df.duration <= max_duration) & # only include fixations of a certain duration
                        (fix_df.active_prob < active_prob_threshold) & # only include fixations with low active prob (i.e. not overlapping with a choice event)
                        (fix_df.trialerror == 0) & # only trials that were completed
                        (fix_df.node_on != fix_df.target)] # exclude fixations after the target has been reached
        
        for trial in trials:
            
            trial_fix = fix_df[(fix_df["trial"] == trial)]
            
            n_fix = trial_fix.shape[0]
            
            fix_values.append(trial_fix.graph_distance.values)
            fix_trial_id.append(np.ones(n_fix)* trial)
            fix_on = np.array(trial_fix["start_time"].values - trial_fix["trial_start_time"].values).astype(int)
            fix_off = np.array(trial_fix["stop_time"].values - trial_fix["trial_stop_time"].values).astype(int)
            
            if type == "mean":
                trial_fix_spikes = np.zeros([n_fix, n_cells])
                for i, ifix in enumerate(trial_fix.iterrows()):
                    trial_fix_spikes[i, ...] = np.nanmean(self.trial_spikes[trial].neural[fix_on[i]:fix_off[i], :], axis=0)
           
            if type == "psth":
                trial_fix_spikes = np.zeros([n_fix, int(2*window_size), n_cells])    
                fix_on = np.array(trial_fix["start_time"].values - trial_fix["trial_start_time"].values).astype(int) - window_size
                fix_off = np.array(trial_fix["start_time"].values - trial_fix["trial_start_time"].values).astype(int) + window_size
                for i, ifix in enumerate(trial_fix.iterrows()):
                    if fix_on[i] < 0:    
                        trial_fix_spikes[i, int(abs(fix_on[i])):, :] = self.trial_spikes[trial].neural[0:fix_off[i], :]
                    elif fix_off[i] > self.trial_spikes[trial].neural.shape[0]:
                        trial_fix_spikes[i, :, :] = self.trial_spikes[trial].neural[fix_on[i]:self.trial_spikes[trial].neural.shape[0]:, :]
                        
                    else:
                        try:
                            trial_fix_spikes[i, ...] = self.trial_spikes[trial].neural[fix_on[i]:fix_off[i], :]
                        except ValueError:
                            print(f"trial {trial}, ifix {i}, fix_on {fix_on[i]}, fix_off {fix_off[i]}, trial length {self.trial_spikes[trial].neural.shape[0]}")
                        
                    
            fix_spikes.append(trial_fix_spikes)
            
        fix_spikes = np.vstack(fix_spikes)
        fix_trial_id = np.hstack(fix_trial_id)
        return dict(spikes=fix_spikes, df=fix_df, trial_id=fix_trial_id)


if __name__ == "__main__":
    fname = '/Volumes/Workspace/Projects/Data/data_Planning_Hu/london/london_092524.nwb'
    region= 'OFC'
    nwb = nwbWrapper(fname, region)
    print(nwb.unitNames)
    print(nwb.trial_df)
    print(nwb.choice_df)
    