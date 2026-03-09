import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from PlanningDynamics import utils
from PlanningDynamics.dataClass import nwbWrapper

def condition_averaged_spikes(X, y):
    conditions = np.unique(y)
    n_conditions = len(conditions)
    _, n_timesteps, n_neurons = X.shape
    condition_average = np.zeros((n_conditions, n_timesteps, n_neurons))
    for i, cond in enumerate(conditions):
        idx = np.where(y == cond)[0]
        condition_average[i, ...] = np.mean(X[idx, ...], axis=0)
    return condition_average, conditions

def get_spikes_value(fname):
    data = nwbWrapper(fname, region="OFC", to_load="all")      
    y_choice = utils.distance_to_value(data.choice_df.graph_distance.values)
    X_choice = data.choice_spikes

    plans = data.get_plan_spikes(type="psth", min_duration=100, active_prob_threshold=0.2)
    y_plan = utils.distance_to_value(plans["df"].graph_distance.values)
    X_plan = plans["spikes"]
    
    choice_avg, choice_cond = condition_averaged_spikes(X_choice, y_choice)
    plan_avg, plan_cond = condition_averaged_spikes(X_plan, y_plan)
    return dict(choice_avg=choice_avg, choice_cond=choice_cond, 
                plan_avg=plan_avg, plan_cond=plan_cond)


def scaledPCA(n_components=20):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', sklearnPCA(n_components=n_components))
    ])
    
def fit_PCA(cond_avg, n_components, bin_size, step_size):
    model = scaledPCA(n_components=n_components)
    n_cond, n_timesteps, n_neurons = cond_avg.shape
    n_bins = int(n_timesteps / step_size)
    X = np.zeros([n_cond, n_bins, n_neurons])
    for i in range(n_bins):
        start = i * step_size
        end = np.minimum(start + bin_size, n_timesteps)
        X[:, i, :] = np.mean(cond_avg[:, start:end, :].astype(float), axis=1)
    X = X.reshape(n_cond * n_bins, n_neurons)
    PCs = model.fit_transform(X)
    PCs = PCs.reshape(n_cond, n_bins, n_components)
    return model, PCs, X, (n_cond, n_bins, n_components)

def projected_response(X, model):
    scaled_X = model["scaler"].transform(X)
    projected_X = model["pca"].transform(scaled_X)
    var_explained = np.diag(np.cov(projected_X.T)) / np.diag(np.cov(scaled_X.T)).sum()    
    return dict(projection=projected_X, var_explained=var_explained)

class PCA:
    def __init__(self, n_components=20, bin_size=150, step_size=10):
        self.n_components = n_components
        self.bin_size = bin_size
        self.step_size = step_size
        self.model = self.scaledPCA()

    def scaledPCA(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca', sklearnPCA(n_components=self.n_components))
        ])

    def reshape_(self, X):
        return X.reshape(self.n_cond, self.n_bins, -1)

    def fit(self, cond_avg):
        self.n_cond, self.n_timesteps, self.n_neurons = cond_avg.shape
        self.n_bins = int(self.n_timesteps / self.step_size)
        X = np.zeros([self.n_cond, self.n_bins, self.n_neurons])
        for i in range(self.n_bins):
            start = i * self.step_size
            end = np.minimum(start + self.bin_size, self.n_timesteps)
            X[:, i, :] = np.mean(cond_avg[:, start:end, :].astype(float), axis=1)
        self.X = X.reshape(self.n_cond * self.n_bins, self.n_neurons)
        PCs = self.model.fit_transform(self.X)
        self.PCs = self.reshape_(PCs)
        #return self.PCs, self.X

    def projected_response(self, X):
        scaled_X = self.model["scaler"].transform(X)
        projected_X = self.model["pca"].transform(scaled_X)
        var_explained = np.diag(np.cov(projected_X.T)) / np.diag(np.cov(scaled_X.T)).sum()    
        return dict(projection=projected_X, var_explained=var_explained)
    
def cross_model_alignment(model1, model2):
        Q = model1.model["pca"].components_
        C = model2.model["pca"].get_covariance()
        lam = model2.model["pca"].singular_values_
        alignment = np.diag(Q @ C @ Q.T).sum()/lam.sum()
        return alignment