import numpy as np
import torch
from scipy.special import factorial, gamma, hyp1f1


class TimeLaggedDataset(torch.utils.data.Dataset):

    """
    the class of dataset design to prepare input data for LaTF training;

    Parameters
    ----------
    pastdata: torch.Tensor;
        the input high-dimensional molecular descriptor at time t;

    futuredata: torch.Tensor;
        the input high-dimensional molecular descriptor at time t+\delta t;

    labeldata: torch.Tensor;
        the initial labels assigned to the input configurations, with a one-to-one correspondence to each configuration

    dataweight: torch.Tensor;
        the data weight associated with each input configuration, enabling the reweighting of training with biased simulation data;

    temperature_weights: torch.Tensor;
        the temperature weight associated with each input configuration, enableing the regularization of data from different 
        temperatures torwards the temperature-steerable exponentially tilted Gaussian prior;
    """
    
    def __init__(self, pastdata, futuredata, labeldata, dataweight, temperature_weights):
        self.past_data = pastdata
        self.future_data = futuredata
        self.future_labels = labeldata
        self.data_weights = dataweight
        self.temperature_weights = temperature_weights
        
    def __len__(self):
        return len(self.past_data)
        
    def __getitem__(self, idx):
        return self.past_data[idx], self.future_labels[idx], self.data_weights[idx], self.temperature_weights[idx]
    
    def update_labels(self, future_labels):
        self.future_labels = future_labels


def _temp_normalization_const(tau=3, dz=2, temp=1):
    """
    quantification of the normalization constant of temperature-steerable exponentially tilted Gaussian; 

    Parameters
    ----------
    tau: float; default = 2;
        the tilting factor tau;

    dz: int; default = 2;
        the dimensionality of the titled Gaussian distribution;

    temp: float; default = 1;
        the temperature-steerable factor to control the radius and variance of the distribution;
    
    Returns
    -------
    a scale represents the normalization factor given the input conditions;

    """
    mu1 = np.sqrt(2) * gamma((dz + 1)/2) / gamma(dz/2)
    term1 = hyp1f1(dz/2, 1/2, 0.5 * tau**2 * temp)
    term2 = hyp1f1((dz+1)/2, 3/2, 0.5 * tau**2 * temp) * mu1 * tau * np.sqrt(temp)
    return term1 + term2



def temp_tilted_prob(z, tau=3, dz=2, temp=1):
    """
    quantification of the probability value of the temperature-steerable exponentially tilted Gaussian;

    Parameters
    ----------
    z: float;
        the input coorindates for the probability distribution;

    tau: float; default = 2;
        the tilting factor tau;

    dz: int; default = 2;
        the dimensionality of the titled Gaussian distribution;

    temp: float; default = 1;
        the temperature-steerable factor to control the radius and variance of the distribution;
    
    Returns
    -------
    a scale represents the probability density at z;
    """

    z_tau = _temp_normalization_const(tau=tau, dz=dz, temp=temp)
    if len(z) > 1:
        norm = np.linalg.norm(z, axis=1)
    else:
        norm = np.linalg.norm(z)
    prob = np.exp(tau*norm)/z_tau * np.exp(-0.5 * norm**2 / temp) / (np.sqrt(2*np.pi))**dz / (np.sqrt(temp))**dz
    return prob



def generate_sample_from_flow(tau, dz, temp, model, num_samples):

    """
    extract samples from temperature-steerable exponentially tilted Gaussian via MC and map to IB latent sample with trained model;

    Parameters
    ----------
    tau: float; default = 2;
        the tilting factor tau;

    dz: int; default = 2;
        the dimensionality of the titled Gaussian distribution;

    temp: float; default = 1;
        the temperature-steerable factor to control the radius and variance of the distribution;

    model: torch.nn.Module;
        the trained LaTF model used for generations;
    
    num_samples: int;
        the number of samples for generation, in the unit of 1e2;

    Returns
    -------
    z_samples: numpy.array;
        the generated samples in the IB latent space;
    """
    
    _cutoff_list = np.linspace(tau*temp, tau*temp+8*temp, 200)
    for cut in _cutoff_list:
        if temp_tilted_prob(z=[cut], tau=tau, dz=dz, temp=temp) < temp_tilted_prob(z=[tau*temp], tau=tau, dz=dz, temp=temp)*1e-4:
            break
    max_prob = temp_tilted_prob([tau*temp], tau=tau, dz=dz, temp=temp)

    samples = []
    for i in range(100):  
        x = np.random.uniform(-cut, cut, num_samples)
        y = np.random.uniform(-cut, cut, num_samples)
        trial_samples = np.concatenate((x[:, None], y[:, None]), axis=1)
        _idx = np.random.uniform(0, max_prob, num_samples) < temp_tilted_prob(trial_samples, tau=tau, dz=dz, temp=temp)
        samples += trial_samples[_idx].tolist()
            
    z_sample = torch.from_numpy(np.array(samples)).float()
    new_samples, _ = model.flow.reverse_sample(z_sample)
    z_samples = new_samples.cpu().detach().numpy()
    return z_samples


def kl_divergence(data_p, data_q, bins=100, epsilon=1e-10):

    """
    quantify the kl divergence between two 2D distribution p and q using data sampled from the distributions;

    Parameters
    ----------

    data_p: numpy.array;
        the data array sampled from distribution p;

    data_q: numpy.array;
        the data array sampled from distribution q;

    bins: int, default = 100;
        the number of bins along each dimension used to grid the 2D space;

    epsilon: float; default = 1e-10;
        the small value used to ensure the data stability;

    Returns
    -------
    kl_div: float
        the scale represents kl divergence;
    """

    hist_p, x_edges, y_edges = np.histogram2d(data_p[:, 0], data_p[:, 1], bins=bins, density=True)
    hist_q, _, _ = np.histogram2d(data_q[:, 0], data_q[:, 1], bins=[x_edges, y_edges], density=True)

    p = hist_p + epsilon
    q = hist_q + epsilon
    p /= np.sum(p)
    q /= np.sum(q)
    kl_div = np.sum(p * np.log(p / q))

    return kl_div