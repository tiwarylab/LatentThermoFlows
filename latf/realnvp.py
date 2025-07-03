import torch


class RealNVP_block(torch.nn.Module):

    """
    create the fundamental building affine block for the RealNVP normalizing flow.

    Parameters
    ----------
    split_mask: torch.Tensor, default = None;
        The mask determines which segments of the input dimensions are alternately transformed at each step of the normalizing flow.
        Has torch.Size([input_dim]), where input_dim denotes the dimensionality of the space used for generative modeling.

    mlp_neurons: int, default = 32;
        The number of neurons in the MLP used within the coupling layer. The scale and shift networks share the same architecture.
        By default, two nonlinear layers with LeakyReLU activation function are employed.

    device: torch device, default = torch.device("cpu")
        The device on which the torch modules are executed.
    """
    
    def __init__(self, split_mask=None, mlp_neurons=32, device=torch.device("cpu")):
        
        super(RealNVP_block, self).__init__()
        self.mask = split_mask
        self.device = device
        self.dim = len(split_mask); self._mlp_neurons = mlp_neurons
        self.scaler_mlp = torch.nn.Sequential(
                                              torch.nn.Linear(self.dim, self._mlp_neurons), torch.nn.LeakyReLU(negative_slope=0.01),
                                              torch.nn.Linear(self._mlp_neurons, self._mlp_neurons), torch.nn.LeakyReLU(negative_slope=0.01),
                                              torch.nn.Linear(self._mlp_neurons, self.dim)
                                              )
        self.shift_mlp = torch.nn.Sequential(
                                              torch.nn.Linear(self.dim, self._mlp_neurons), torch.nn.LeakyReLU(negative_slope=0.01),
                                              torch.nn.Linear(self._mlp_neurons, self._mlp_neurons), torch.nn.LeakyReLU(negative_slope=0.01),
                                              torch.nn.Linear(self._mlp_neurons, self.dim)
                                              )
        self.weights = torch.nn.Parameter(torch.Tensor(self.dim))
        
        
    def forward(self, x):

        """
        forward scale-and-shift transformation through an affine layer and quantification of associated Jacobian determinant.

        Parameters
        ----------
        x: torch.Tensor
            The input data for the affine layer, should share the same dimension as split_mask.
        """

        x_mask = x * self.mask
        _s = self.scaler_mlp(x_mask) * self.weights
        _x = x_mask + (torch.exp(_s) * x + self.shift_mlp(x_mask)) * (1-self.mask)
        log_detabs_jacob = torch.sum(_s * (1-self.mask), dim=-1).to(self.device)
        return _x, log_detabs_jacob
    

    def inverse(self, _x):

        """
        backward (inverse) scale-and-shift transformation through an affine layer and quantification of associated Jacobian determinant.

        Parameters
        ----------
        _x: torch.Tensor
            The input data for the affine layer, should share the same dimension as split_mask.
        """

        _x_mask = _x * self.mask
        _s = self.scaler_mlp(_x_mask) * self.weights
        x = _x_mask + (1-self.mask) * (_x - self.shift_mlp(_x_mask)) / torch.exp(_s)
        inv_log_detabs_jacob = torch.sum(-_s * (1-self.mask), dim=-1).to(self.device)
        return x, inv_log_detabs_jacob
    


class RealNVP(torch.nn.Module):

    """
    create RealNVP normalizing flow model consisting sequence of affine blocks.

    Parameters
    ----------
    split_mask: torch.Tensor, default = None;
        The mask determines which segments of the input dimensions are alternately transformed at each step of the normalizing flow.
        Has torch.Size([input_dim]), where input_dim denotes the dimensionality of the space used for generative modeling.
    
    num_layers: int, default = 8;
        The number of affine coupling layers used to conduct the bijective flow transformation.
   
     mlp_neurons: int, default = 32;
        The number of neurons in the MLP used within the coupling layer. The scale and shift networks share the same architecture.
        By default, two nonlinear layers with LeakyReLU activation function are employed.

    train_boolearn: boolean, default = True;
        The parameter controls if the normalizing flow is trained and optimized together with SPIB.

    device: torch device, default = torch.device("cpu")
        The device on which the torch modules are executed.
    """
    
    def __init__(self, split_mask=None, num_layers=8, mlp_neurons=32, train_boolean=True, device=torch.device("cpu")):
        super(RealNVP, self).__init__()
        self.dim = split_mask.shape[-1]
        self.device = device
        self.training = train_boolean
        self.model = torch.nn.ModuleList([RealNVP_block(split_mask=split_mask[i], mlp_neurons=mlp_neurons, device=self.device) for i in range(num_layers)])
    
    
    def forward(self, x):
        
        log_det = torch.zeros(len(x)).to(self.device)
        for layer in reversed(self.model):
            x, inv_log_detabs_jacob = layer.inverse(x)
            log_det += inv_log_detabs_jacob
            
        return x, log_det
    
    
    def reverse_sample(self, z):
        
        log_det = torch.zeros(len(z)).to(self.device)
        for layer in self.model:
            z, log_detabs_jacob = layer(z)
            log_det += log_detabs_jacob
        return z, log_det