import torch
from torch import nn
import numpy as np
import os
import time
import torch.nn.functional as F
from realnvp import *
    
    
class LaTF(torch.nn.Module):

    """
    class of Latent Thermodynamic Flows (LaTF) model

    Parameters
    ----------
    output_dim: int, default = None;
        The number of initial states clustered upon physical order parameters;
        The real output dimension will be updated on the fly (short-lived states are emerged into long-lived states);

    data_shape: list, defaut = None;
        List consists of the shape of input molecular descriptors; will be used to determine the number of neurons in the encoder inpu layer;

    encoder_type: str, default = 'Nonlinear';
        If set 'Nonlinear' then nonlinear encoder will be utilized, otherwise, encoder will be set as a linear layer;

    u_dim: int, default = 2;
        Dimension of the latent Information Bottleneck space, 2-dimension is recommended;

    lagtime: int, default = 1;
        Lag time delta t between the input molecular descriptor and output state label; only used for printing;

    beta: float, default = 1e-3;
        The weight of regularization term (including prior and Jacobian) in the loss function;

    learning_rate: float; default = 1e-3;
        The learning rate adopted by the optimizer to train the LaTF model;

    lr_scheduler_gamma: float, default = 1.0;
        The parameter used to ajust learning rate decay by LambdaLR;

    neuron_num1: int, default = 64;
        The number of neurons in the encoder layers, the encoder architecture is fixed to consist of two layers;

    neuron_num2: int, default = 64;
        The number of neurons in the decoder layers, the decoder architecture is fixed to consist of two nonlinear layers;

    flow_layers: int, default = 8;
        The number of affine coupling layers in the RealNVP normalizing flow;

    flow_neurons: int, default = 16;
        The number of neurons in the scaling and shifting mlps of the RealNVP normalizing flow;

    flow_split_mask: torch.Tensor, default = None;
        The mask determines which segments of the input dimensions are alternately transformed at each step of the normalizing flow;
        Has torch.Size([input_dim]), where input_dim denotes the dimensionality of the space used for generative modeling;

    tilted_tau: float; defaut = 3.0;
        The tilting factor tau in the exponentially tilted Gaussian prior;

    UpdateLabel: boolean, default = True;
        Determine if the labels of input configurations will be updated to their future-transition state label;

    device: torch device, default = torch.device("cpu");
        The device on which the torch modules are executed.
    """

    def __init__(self, output_dim=None, data_shape=None, encoder_type='Nonlinear', u_dim=2, lagtime=1, beta=1e-3,
                 learning_rate=1e-3, lr_scheduler_gamma=1.0, neuron_num1=64, neuron_num2=64, flow_layers=8,
                 flow_neurons=16, flow_split_mask=None, tilted_tau=3.0, UpdateLabel=True, device=torch.device("cpu")):

        super(LaTF, self).__init__()
        if encoder_type == 'Nonlinear':
            self.encoder_type = 'Nonlinear'
        else:
            self.encoder_type = 'Linear'

        self.u_dim = u_dim
        self.lagtime = lagtime
        self.beta = beta

        self.learning_rate = learning_rate
        self.lr_scheduler_gamma = lr_scheduler_gamma

        self.output_dim = output_dim

        self.neuron_num1 = neuron_num1
        self.neuron_num2 = neuron_num2

        self.flow_layers = flow_layers
        self.flow_neurons = flow_neurons
        self.tilted_tau = tilted_tau

        self.data_shape = data_shape

        self.UpdateLabel = UpdateLabel

        self.eps = 1e-10
        self.device = device

        # The collected relative_state_population_change. First dimension contains the step, second dimension the change.
        self.relative_state_population_change_history = []
        # The collected train loss. First dimension contains the step, second dimension the loss. Initially empty.
        self.train_loss_history = []
        # The collected test loss. First dimension contains the step, second dimension the loss. Initially empty.
        self.test_loss_history = []
        # The collected number of states. [ refinement id, number of epoch used for this refinement, number of states ]
        self.convergence_history = []

        # torch buffer, these variables will not be trained
        self.register_buffer('representative_inputs', torch.eye(self.output_dim, np.prod(self.data_shape), device=device, requires_grad=False))

        # create an idle input for calling representative-weights
        # torch buffer, these variables will not be trained
        self.register_buffer('idle_input', torch.eye(self.output_dim, self.output_dim, device=device, requires_grad=False))

        # representative weights
        self.vampprior_weights = nn.Sequential(
            nn.Linear(self.output_dim, 1, bias=False),
            nn.Softmax(dim=0)).to(self.device)

        self.encoder = self._encoder_init()

        if self.encoder_type == 'Nonlinear':
            self.encoder_mean = nn.Linear(self.neuron_num1, self.u_dim).to(self.device)
        else:
            self.encoder_mean = nn.Linear(np.prod(self.data_shape), self.u_dim).to(self.device)

        self.encoder_logvar = nn.Parameter(torch.tensor([0.0]), requires_grad=True).to(self.device)

        self.decoder = self._decoder_init()

        self.decoder_output = nn.Sequential(
            nn.Linear(self.neuron_num2, self.output_dim),
            nn.LogSoftmax(dim=1)).to(self.device)
        
        if flow_split_mask == None:
            self._split_mask = torch.nn.functional.one_hot(torch.tensor([i%2 for i in range(flow_layers)])).float().to(self.device)
        else:
            self._split_mask = flow_split_mask
        self.flow = RealNVP(split_mask=self._split_mask, num_layers=self.flow_layers, mlp_neurons=self.flow_neurons, device=self.device)

    def _encoder_init(self):

        modules = []
        modules += [nn.Linear(np.prod(self.data_shape), self.neuron_num1)]
        modules += [nn.ReLU()]
        for _ in range(1):
            modules += [nn.Linear(self.neuron_num1, self.neuron_num1)]
            modules += [nn.ReLU()]
        # modules += [nn.LayerNorm(self.neuron_num1)]
        return nn.Sequential(*modules).to(self.device)

    def _decoder_init(self):
        # cross-entropy MLP decoder
        # output the probability of future state
        # modules = [nn.LayerNorm(self.u_dim)] 
        modules = [nn.Linear(self.u_dim, self.neuron_num2)]
        modules += [nn.ReLU()]
        for _ in range(1):
            modules += [nn.Linear(self.neuron_num2, self.neuron_num2)]
            modules += [nn.ReLU()]

        return nn.Sequential(*modules).to(self.device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu

    def encode(self, inputs):

        if self.encoder_type == 'Nonlinear':
            enc = self.encoder(inputs)
            u_mean = self.encoder_mean(enc)
        else:
            u_mean = self.encoder_mean(inputs)

        u_logvar = self.encoder_logvar

        return u_mean, u_logvar

    def decode(self, u):
        dec = self.decoder(u)
        outputs = self.decoder_output(dec)

        return outputs

    def forward(self, data):
        inputs = torch.flatten(data, start_dim=1)

        u_mean, u_logvar = self.encode(inputs)

        u_sample = self.reparameterize(u_mean, u_logvar)
        
        outputs = self.decode(u_sample)
        
        if self.flow.training:
            flow_z, log_det = self.flow(u_sample)
            return outputs, u_sample, u_mean, u_logvar, flow_z, log_det
        else:
            return outputs, u_sample, u_mean, u_logvar


    def calculate_loss(self, data_inputs, data_targets, data_weights, beta, temperature_weights):

        """
        quantify the loss function for LaTF training;
            if flow.training is True, the joint objective function is used to optimize both encoder, decoder and the normalizing flow;
            if flow.training is False, only the encoder and decoder are optimized based on the original SPIB objective;
        
        Parameters
        ----------
        data_input: torch.Tensor;
            the input data for the LaTF, which will go through encoder and sequently the normalizing flow;

        data_targets: torch.Tensor;
            the future state label where the input configuration is going to transite into;

        data_weights: torch.Tensor;
            the data weight associated with each input configuration, enabling the reweighting of training with biased simulation data;

        beta: torch.Tensor;
            a single scale to trade off between the reconstruction error and regularization error;

        temperature_weights: torch.Tensor;
            the temperature weight associated with each input configuration, enableing the regularization of data from different 
            temperatures torwards the temperature-steerable exponentially tilted Gaussian prior;

        Returns
        -------
        loss: torch.Tensor;
            the total loss value quantified on the data with gradient;
        reconstruction_error: torch.Tensor;
            the accuracy of prediction of future state labels, without gradient on the model parameters;
        kl_loss: torch.Tensor;
            the regularization loss wrt the prior, without gradient on the model parameters;

        """

        if self.flow.training:
            outputs, u_sample, u_mean, u_logvar, flow_z, log_det = self.forward(data_inputs)
            log_p = self.log_p(flow_z, data_weights, temperature_weights)
            likelihood_loss = torch.mean(-log_p)
            det_loss = torch.mean((-log_det)*data_weights[:, None])
            kl_loss = likelihood_loss + det_loss
            reconstruction_error = torch.mean(torch.sum(-data_targets*outputs, dim=1)*data_weights)
        
        else:    
            outputs, u_sample, u_mean, u_logvar = self.forward(data_inputs)
            log_p = self.log_p(u_sample, data_weights, temperature_weights)
            log_q = -0.5 * torch.sum(u_logvar + torch.pow(u_sample-u_mean, 2) / torch.exp(u_logvar), dim=1)
            kl_loss = torch.mean((log_q-log_p)*data_weights)
            reconstruction_error = torch.mean(torch.sum(-data_targets*outputs, dim=1)*data_weights)

        loss = reconstruction_error + beta * kl_loss
        return loss, reconstruction_error.detach().cpu().data, kl_loss.detach().cpu().data
    

    def log_p (self, z, data_weights, temperature_weights):
        
        """
        majorly used to quantify the likelihood of the encoded / encoded-flowed transformed input on the prior distribution;

        Parameters
        ----------
        z: torch.Tensor; 
            encoded / encoded-flowed transformed input value in the prior space;
        
        data_weights: torch.Tensor;
            the data weight associated with each input configuration, enabling the reweighting of training with biased simulation data;

        temperature_weights: torch.Tensor;
            the temperature weight associated with each input configuration, enableing the regularization of data from different 
            temperatures torwards the temperature-steerable exponentially tilted Gaussian prior;

        Returns
        -------
        log_p: torch.Tensor;
            the log-probability of the encoded / encoded-flowed transformed data on the prior;
        """
    
        # get representative_z
        # shape: [output_dim, z_dim]
        representative_z_mean, representative_z_logvar = self.get_vampprior_mean_var()
        # get representative weights
        # shape: [output_dim, 1]
        w = self.vampprior_weights(self.idle_input)

        # expand z
        # shape: [batch_size, z_dim]
        z_expand = z.unsqueeze(1)
        # print("z shape: {}; z_expand shape: {};".format(z.shape, z_expand.shape))

        representative_mean = representative_z_mean.unsqueeze(0)
        representative_logvar = representative_z_logvar.unsqueeze(0)

        # representative log_q
        representative_log_q = -0.5 * torch.sum(representative_logvar + (z_expand-representative_mean)**2
                                        / torch.exp(representative_logvar), dim=2)
        if self.flow.training:
            norm_z = torch.sum((z_expand)**2, dim=2)
            log_p = torch.sum((-norm_z * 0.5 * temperature_weights[:, None] + self.tilted_tau * torch.sqrt(norm_z))*data_weights[:, None], dim=1)
        else:
            log_p = torch.sum((torch.log(torch.exp(representative_log_q)@w + self.eps))*data_weights[:, None], dim=1)

        return log_p


    @torch.no_grad()
    def get_vampprior_mean_var(self):
        
        """
        get the mean value and variance value for the variational mixture of posterior prior from the pseudo-inputs;
        """
        
        # calculate representative_means
        X = self.representative_inputs

        # calculate representative_z
        representative_z_mean, representative_z_logvar = self.encode(X)  # C x M
        
        if self.flow.training:
            z_sample = self.reparameterize(representative_z_mean, representative_z_logvar)
            flow_z_mean, _ = self.flow(z_sample)
            return flow_z_mean, torch.ones(flow_z_mean.shape).to(self.device)
        else:
            return representative_z_mean, representative_z_logvar
        

    def reset_vampprior_weights(self, representative_inputs):

        """
        update the linear combination coefficient weight of variational mixture of posterior prior;
        the weight needs to be re-initialized and its number needs to be reduced to be consistent with state number;

        Parameters
        ----------
        representative_inputs: torch.Tensor;
            the pseudo-inputs in the input data used to get the VampPrior mean and variance parameters;

        """

        # reset the nuber of representative inputs
        self.output_dim = representative_inputs.shape[0]

        # reset representative weights
        self.idle_input = torch.eye(self.output_dim, self.output_dim, device=self.device, requires_grad=False)

        self.vampprior_weights = nn.Sequential(
            nn.Linear(self.output_dim, 1, bias=False),
            nn.Softmax(dim=0))
        self.vampprior_weights[0].weight = nn.Parameter(torch.ones([1, self.output_dim], device=self.device))

        # reset representative inputs
        self.representative_inputs = representative_inputs.clone().detach()


    @torch.no_grad()
    def init_vampprior_inputs(self, inputs, labels):
        """
        randomly initialize the VampPrior pseudo-inputs from the input data for each state;
        
        Parameters
        ----------
        inputs: torch.Tensor;
            the input high-dimensional molecular descriptors;

        labels: torch.Tensor;
            the associated labels with the input molecular configuration; 

        Returns
        -------
        representative_inputs: torch.Tensor;
            the VampPrior pseudo-inputs from the input data;
        
        """
        state_population = labels.sum(dim=0).cpu()

        # randomly pick up one sample from each initlal state as the initial guess of representative-inputs
        representative_inputs = []

        for i in range(state_population.shape[-1]):
            if state_population[i] > 0:
                index = np.random.randint(0, state_population[i])
                representative_inputs += [inputs[labels[:, i].bool()][index].reshape(1, -1)]
            else:
                # randomly select one sample as the representative input
                index = np.random.randint(0, inputs.shape[0])
                representative_inputs += [inputs[index].reshape(1, -1)]

        # print(representative_inputs)
        representative_inputs = torch.cat(representative_inputs, dim=0)

        self.reset_vampprior_weights(representative_inputs.to(self.device))

        return representative_inputs
    


    @torch.no_grad()
    def update_model(self, inputs, data_weights, train_data_labels, test_data_labels, batch_size, threshold=0):

        """
        with the updated labels for inputs, relocate the pseudo-inputs for the VampPrior as the geometric center of each state in IB space;
        reinitlize and adjust the number of linear combination coefficient for the VampPrior according to the number of states;
        remove the future-state labels with zero population;
        shrink the output layer to be consistent with the current number of states while copy the associated weights and bias;

        Parameters
        ----------
        inputs: torch.Tensor;
            all the (past) data used to relocate the pseudo-inputs for VampPrior;

        data_weights: torch.Tensor;
            the data weight associated with each input configuration, enabling the reweighting of training with biased simulation data;

        train_data_labels: torch.Tensor;
            the one-hot vector with shape [num_frames, num_states] corresponds to the training data;

        test_data_labels: torch.Tensor;
            the one-hot vector with shape [num_frames, num_states] corresponds to the testing data;

        batch_size: int;
            the training batch size;

        threshod: float; default = 0;
            the state population threshod used to decide labels needed to be removed;

        Returns
        -------
        train_data_labels, test_data_labels :  torch.Tensor;
            the updated state label for either train data or test data;

        """
        mean_rep = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(self.device)
            u_mean, u_logvar = self.encode(batch_inputs)
            mean_rep += [u_mean]

        mean_rep = torch.cat(mean_rep, dim=0)

        state_population = train_data_labels.sum(dim=0).float() / train_data_labels.shape[0]

        # ignore states whose state_population is smaller than threshold to speed up the convergence
        # By default, the threshold is set to be zero
        train_data_labels = train_data_labels[:, state_population > threshold]
        test_data_labels = test_data_labels[:, state_population > threshold]

        # save new guess of representative-inputs
        representative_inputs = []

        for i in range(train_data_labels.shape[-1]):
            weights = data_weights[train_data_labels[:, i].bool()].reshape(-1, 1)
            center_z = ((weights * mean_rep[train_data_labels[:, i].bool()]).sum(dim=0) / weights.sum()).reshape(1, -1)

            # find the one cloest to center_z as representative-inputs
            dist = torch.square(mean_rep - center_z).sum(dim=-1)
            index = torch.argmin(dist)
            representative_inputs += [inputs[index].reshape(1, -1)]

        representative_inputs = torch.cat(representative_inputs, dim=0)

        ## reset the vampprior linear combination coefficient;
        self.reset_vampprior_weights(representative_inputs)

        # record the old parameters of the output layer for states with certain populations;
        w = self.decoder_output[0].weight[state_population > threshold]
        b = self.decoder_output[0].bias[state_population > threshold]

        # reset the dimension of the output layer;
        self.decoder_output = nn.Sequential(
            nn.Linear(self.neuron_num2, self.output_dim),
            nn.LogSoftmax(dim=1))

        self.decoder_output[0].weight = nn.Parameter(w.to(self.device))
        self.decoder_output[0].bias = nn.Parameter(b.to(self.device))

        return train_data_labels, test_data_labels


    @torch.no_grad()
    def update_labels(self, inputs, batch_size):
        """
        relabel the input configuration with the label of its future state that has the highest transition probability;
        """
        if self.UpdateLabel:
            labels = []

            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i+batch_size]

                # pass through VAE
                u_mean, u_logvar = self.encode(batch_inputs)
                log_prediction = self.decode(u_mean)

                # label = p/Z
                labels += [log_prediction.exp()]

            labels = torch.cat(labels, dim=0)
            max_pos = labels.argmax(1)
            labels = F.one_hot(max_pos, num_classes=self.output_dim)

            return labels


    def fit(self, train_dataset, test_dataset, batch_size=128, flow_train=True, tolerance=0.001, patience=5, refinements=15,\
            output_path='./latf_output/latf', beta=1e-3, optimizer='Adam', mask_threshold=0):
        
        """ Training of a LaTF model with data.
            train one epoch -> quantify changes of loss -> smaller than tolerance for patience times ->
            update models and relabel input data -> updating for refinement times -> output self-consistent model;

        Parameters
        ----------
        train_dataset: latf.utils.TimeLaggedDataset;
            The data to use for training. Should yield a tuple of batches representing
            instantaneous samples, time-lagged labels, sample weights and temperature weights.

        test_dataset: latf.utils.TimeLaggedDataset;
            The data to use for test. Should yield a tuple of batches representing
            instantaneous samples, time-lagged labels, sample weights and temperature weights.

        batch_size: int, default = 128;

        flow_train: boolean, default = True;
        The parameter controls if the normalizing flow is trained and optimized together with encoder and decoder.

        tolerance: float, default = 0.001
            tolerance of loss change for measuring the convergence of the training

        patience: int, default = 5
            Number of epochs with the change of the state population smaller than the threshold
            after which this iteration of training finishes.

        refinements: int, default = 15
            Number of refinements, with which the model updating and input relabeling happen.

        output_path: str, default= './latf_output/latf'
            The output address in which the training log file and updated models are saved.

        beta: float, default = 1e-3;
            The weight balances between reconstruction loss and regularization loss;

        optimizer: str, default = 'Adam';
            The type of optimizer used for training. Should be chosen from {'Adam', 'SGD', 'RMSprop'}

        mask_threshold: float, default=0
            Minimum probability for checking convergence.

        Returns
        -------
        self : LaTF
            Reference to self.
        """

        self.train()
        self.flow.training = flow_train
        self.output_path = output_path

        # data preparation
        # Specify BatchSampler as sampler to speed up dataloader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(train_dataset), batch_size, False), batch_size=None)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(test_dataset), batch_size, False), batch_size=None)

        # use the training set to initialize the pseudo-inputs
        self.init_vampprior_inputs(train_dataset.past_data, train_dataset.future_labels)

        self.optimizer_types = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, 'RMSprop': torch.optim.RMSprop}
        if optimizer not in self.optimizer_types.keys():
            raise ValueError(f"Unknown optimizer type, supported types are {self.optimizer_types.keys()}")
        else:
            self._optimizer = self.optimizer_types[optimizer](self.parameters(), lr=self.learning_rate)

        lr_lambda = lambda epoch: self.lr_scheduler_gamma ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        start = time.time()
        log_path = self.output_path + '_train.log'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        IB_path = self.output_path + "cpt/IB"
        os.makedirs(os.path.dirname(IB_path), exist_ok=True)

        step = 0
        update_times = 0
        unchanged_epochs = 0
        epoch = 0

        # initial state population
        state_population0 = (torch.sum(train_dataset.future_labels, dim=0).float() / train_dataset.future_labels.shape[0]).cpu()
        train_epoch_loss0 = 0

        while True:

            for batch_inputs, batch_outputs, batch_weights, batch_temperatures in train_dataloader:
                step += 1

                loss, reconstruction_error, kl_loss = self.calculate_loss(batch_inputs, batch_outputs, batch_weights, beta, batch_temperatures)

                # if self.flow.training:
                #     loss = torch.clamp(loss, max=50.0) 
                # Stop if NaN is obtained
                if(torch.isnan(loss).any()):
                    print("Loss is nan!")
                    raise ValueError

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            epoch += 1

            scheduler.step()
            if self.lr_scheduler_gamma < 1:
                print("Update lr to %f" % (self._optimizer.param_groups[0]['lr']))
                print("Update lr to %f" % (self._optimizer.param_groups[0]['lr']), file=open(log_path, 'a'))

            with torch.no_grad():
                train_time = time.time() - start

                train_epoch_loss = 0
                train_epoch_kl_loss = 0
                train_epoch_reconstruction_error = 0

                train_epoch_loss += loss.detach().cpu().data * len(batch_inputs)
                train_epoch_kl_loss += kl_loss  * len(batch_inputs)
                train_epoch_reconstruction_error += reconstruction_error * len(batch_inputs)

                self.train_loss_history += [[step, loss.detach().cpu().data.numpy()]]

                weight_sum = train_dataset.data_weights.sum().cpu()
                train_epoch_loss /= weight_sum
                train_epoch_kl_loss /= weight_sum
                train_epoch_reconstruction_error /= weight_sum

                print(
                    "Epoch %i:\tTime %f s\nLoss (train) %f\tkl loss (train): %f\t"
                    "Reconstruction loss (train) %f" % (
                        epoch, train_time, train_epoch_loss, train_epoch_kl_loss, train_epoch_reconstruction_error))
                print(
                    "Epoch %i:\tTime %f s\nLoss (train) %f\tlikelihood loss (train): %f\t"
                    "Reconstruction loss (train) %f" % (
                        epoch, train_time, train_epoch_loss, train_epoch_kl_loss,
                        train_epoch_reconstruction_error), file=open(log_path, 'a'))
                


                test_epoch_loss = 0
                test_epoch_kl_loss = 0
                test_epoch_reconstruction_error = 0
                for batch_inputs, batch_outputs, batch_weights, batch_temperatures in test_dataloader:

                    loss, reconstruction_error, kl_loss = self.calculate_loss(batch_inputs, batch_outputs, batch_weights, beta, batch_temperatures)
                    
                    weight_sum = batch_weights.sum().cpu()
                    test_epoch_loss += loss.cpu().data * len(batch_inputs)
                    test_epoch_kl_loss += kl_loss * len(batch_inputs)
                    test_epoch_reconstruction_error += reconstruction_error * len(batch_inputs)

                weight_sum = test_dataset.data_weights.sum().cpu()
                test_epoch_loss /= weight_sum
                test_epoch_kl_loss /= weight_sum
                test_epoch_reconstruction_error /= weight_sum

                print(
                    "Loss (test) %f\tkl loss (test): %f\t"
                    "Reconstruction loss (test) %f" % (
                        test_epoch_loss, test_epoch_kl_loss, test_epoch_reconstruction_error))
                print(
                    "Loss (test) %f\kl loss (test): %f\t"
                    "Reconstruction loss (test) %f" % (
                        test_epoch_loss, test_epoch_kl_loss, test_epoch_reconstruction_error), file=open(log_path, 'a'))

                self.test_loss_history += [[step, test_epoch_loss.cpu().data.numpy()]]

            print('training total loss change=%f' % (train_epoch_loss - train_epoch_loss0))
            print('training total loss change=%f' % (train_epoch_loss - train_epoch_loss0), file=open(log_path, 'a'))

            # check convergence
            new_train_data_labels = self.update_labels(train_dataset.future_data, batch_size)

            # save the state population
            state_population = (torch.sum(new_train_data_labels, dim=0).float()/new_train_data_labels.shape[0]).cpu()

            print('State population:')
            print('State population:', file=open(log_path, 'a'))
            print(state_population.numpy())
            print(state_population.numpy(), file=open(log_path, 'a'))

            # print the relative state population change
            mask = (state_population0 > mask_threshold)
            relative_state_population_change = torch.sqrt(
                torch.square((state_population - state_population0)[mask] / state_population0[mask]).mean())

            print('Relative state population change=%f' % relative_state_population_change)
            print('Relative state population change=%f' % relative_state_population_change, file=open(log_path, 'a'))

            self.relative_state_population_change_history += [[step, relative_state_population_change.numpy()]]

            # update state_population
            state_population0 = state_population

            # check whether the change of the training loss is smaller than the tolerance
            if torch.abs(train_epoch_loss - train_epoch_loss0) < tolerance:
                unchanged_epochs += 1

                if unchanged_epochs > patience:
                    # save model
                    torch.save({'refinement': update_times+1,
                                'state_dict': self.state_dict()},
                               IB_path + '_%d_cpt.pt' % (update_times+1))
                    torch.save({'optimizer': self._optimizer.state_dict()},
                               IB_path + '_%d_optim_cpt.pt' % (update_times+1))

                    # check whether only one state is found
                    if torch.sum(state_population>0)<2:
                        print("Only one metastable state is found!")
                        raise ValueError

                    # Stop only if update_times >= min_refinements
                    if self.UpdateLabel and update_times < refinements:

                        train_data_labels = new_train_data_labels
                        test_data_labels = self.update_labels(test_dataset.future_data, batch_size)
                        train_data_labels = train_data_labels.to(self.device)
                        test_data_labels = test_data_labels.to(self.device)

                        update_times += 1
                        print("Update %d\n" % (update_times))
                        print("Update %d\n" % (update_times), file=open(log_path, 'a'))

                        # update the model, and reset the representative-inputs
                        train_data_labels, test_data_labels = self.update_model(train_dataset.past_data,
                                                                                train_dataset.data_weights,
                                                                                train_data_labels, test_data_labels,
                                                                                batch_size, mask_threshold)

                        train_dataset.update_labels(train_data_labels)
                        test_dataset.update_labels(test_data_labels)

                        # initial state population
                        state_population0 = (torch.sum(train_data_labels, dim=0).float() / train_data_labels.shape[0]).cpu()

                        # reset the optimizer and scheduler
                        scheduler.last_epoch = -1

                        # save the history [ refinement id, number of epoch used for this refinement, number of states ]
                        self.convergence_history += [[update_times, epoch, self.output_dim]]

                        # reset epoch and unchanged_epochs
                        epoch = 0
                        unchanged_epochs = 0

                    else:
                        break

            else:
                unchanged_epochs = 0

            train_epoch_loss0 = train_epoch_loss

        # output the saving path
        total_training_time = time.time() - start
        print("Total training time: %f" % total_training_time)
        print("Total training time: %f" % total_training_time, file=open(log_path, 'a'))

        self.eval()

        # label update
        if self.UpdateLabel:
            train_data_labels = self.update_labels(train_dataset.future_data, batch_size)
            test_data_labels = self.update_labels(test_dataset.future_data, batch_size)

            # update the model, and reset the representative-inputs
            train_data_labels, test_data_labels = self.update_model(train_dataset.past_data, train_dataset.data_weights,
                                                                    train_data_labels, test_data_labels, batch_size)

            train_dataset.update_labels(train_data_labels)
            test_dataset.update_labels(test_data_labels)

        # save model
        torch.save({'step': step,
                    'state_dict': self.state_dict()},
                   IB_path + '_final_cpt.pt')
        torch.save({'optimizer': self._optimizer.state_dict()},
                   IB_path + '_final_optim_cpt.pt')
        
        return self


    @torch.no_grad()
    def transform(self, data, batch_size=128, to_numpy=False):

        """ Transforms data through the instantaneous or time-shifted network lobe.

        Parameters
        ----------
        data : numpy ndarray or torch tensor
            The data to transform.

        batch_size : int, default=128
        
        to_numpy: bool, default=True
            Whether to convert torch tensor to numpy array.
        Returns
        -------
        List of numpy array or torch tensor containing transformed data.
        """
        self.eval()

        if isinstance(data, torch.Tensor):
            inputs = data
        else:
            inputs = torch.from_numpy(data.copy()).float()

        all_prediction = []
        all_u_mean = []
        all_u_logvar = []

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(self.device)

            u_mean, u_logvar = self.encode(batch_inputs)

            log_prediction = self.decode(u_mean)

            all_prediction += [log_prediction.exp().cpu()]
            all_u_logvar += [u_logvar.cpu()]
            all_u_mean += [u_mean.cpu()]

        all_prediction = torch.cat(all_prediction, dim=0)
        all_u_logvar = torch.cat(all_u_logvar, dim=0)
        all_u_mean = torch.cat(all_u_mean, dim=0)

        labels = all_prediction.argmax(1)

        if to_numpy:
            return labels.numpy().astype(np.int32), all_prediction.numpy().astype(np.double), \
                   all_u_mean.numpy().astype(np.double), all_u_logvar.numpy().astype(np.double)
        else:
            return labels, all_prediction, all_u_mean, all_u_logvar








