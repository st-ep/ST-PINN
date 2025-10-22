import time
import numpy as np
import torch
import torch.nn as nn
from source.pdes import Diffusion_sorption, Boundary
from source.utilities import mean_squared_error, relative_error, set_random_seed, get_device

set_random_seed(1234)


class NeuralNetReLU(nn.Module):
    """Custom Neural Network with ReLU output activation (specific for DiffSorb)"""

    def __init__(self, *inputs, layers, device=None):
        super(NeuralNetReLU, self).__init__()

        self.layers = layers
        self.num_layers = len(self.layers)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Input normalization
        if len(inputs) == 0:
            in_dim = self.layers[0]
            self.X_mean = torch.zeros([1, in_dim], dtype=torch.float32, device=self.device)
            self.X_std = torch.ones([1, in_dim], dtype=torch.float32, device=self.device)
        else:
            X = np.concatenate(inputs, 1)
            self.X_mean = torch.tensor(X.mean(0, keepdims=True), dtype=torch.float32, device=self.device)
            self.X_std = torch.tensor(X.std(0, keepdims=True), dtype=torch.float32, device=self.device)

        # Initialize parameters
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.gammas = nn.ParameterList()

        for l in range(0, self.num_layers - 1):
            in_dim = self.layers[l]
            out_dim = self.layers[l + 1]

            W = torch.randn(in_dim, out_dim, dtype=torch.float32, device=self.device)
            b = torch.zeros(1, out_dim, dtype=torch.float32, device=self.device)
            g = torch.ones(1, out_dim, dtype=torch.float32, device=self.device)

            self.weights.append(nn.Parameter(W))
            self.biases.append(nn.Parameter(b))
            self.gammas.append(nn.Parameter(g))

    def forward(self, *inputs):
        H = torch.cat(inputs, 1)
        H = (H - self.X_mean) / self.X_std

        for l in range(0, self.num_layers - 1):
            W = self.weights[l]
            b = self.biases[l]
            g = self.gammas[l]

            V = W / torch.norm(W, dim=0, keepdim=True)
            H = torch.matmul(H, V)
            H = g * H + b

            if l < self.num_layers - 2:
                H = torch.tanh(H)

        # Apply ReLU to output
        Y = torch.relu(torch.split(H, 1, dim=1))

        return Y


class SelfTrainingPINN:
    """Self-Training Physics-Informed Neural Network for 1D Diffusion-Sorption Equation"""

    def __init__(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_sample, t_sample, x_data,
                 t_data, u_data, x_test, t_test, u_test, batch_size, layers, log_path, update_freq, max_rate,
                 stab_coeff, device=None):

        # Device setup
        self.device = device if device is not None else get_device()

        # Convert data to torch tensors
        self.x_init = torch.tensor(x_init, dtype=torch.float32, device=self.device)
        self.t_init = torch.tensor(t_init, dtype=torch.float32, device=self.device)
        self.u_init = torch.tensor(u_init, dtype=torch.float32, device=self.device)

        self.x_l_bound = torch.tensor(x_l_bound, dtype=torch.float32, device=self.device)
        self.x_r_bound = torch.tensor(x_r_bound, dtype=torch.float32, device=self.device)
        self.t_bound = torch.tensor(t_bound, dtype=torch.float32, device=self.device)

        self.x_sample = torch.tensor(x_sample, dtype=torch.float32, device=self.device)
        self.t_sample = torch.tensor(t_sample, dtype=torch.float32, device=self.device)

        self.x_data = torch.tensor(x_data, dtype=torch.float32, device=self.device)
        self.t_data = torch.tensor(t_data, dtype=torch.float32, device=self.device)
        self.u_data = torch.tensor(u_data, dtype=torch.float32, device=self.device)

        self.x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)
        self.t_test = torch.tensor(t_test, dtype=torch.float32, device=self.device)
        self.u_test = torch.tensor(u_test, dtype=torch.float32, device=self.device)

        # Pseudo-label tracking
        self.flag_pseudo = np.zeros(shape=x_sample.shape, dtype=np.int32)

        # Dynamic point arrays
        self.x_eqns = None
        self.t_eqns = None
        self.x_pseudo = None
        self.t_pseudo = None
        self.u_pseudo = None

        self.layers = layers
        self.log_path = log_path
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.max_rate = max_rate
        self.stab_coeff = stab_coeff

        # Initialize neural network (with ReLU output)
        self.net = NeuralNetReLU(x_sample, t_sample, layers=self.layers, device=self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def compute_loss(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound,
                     x_data, t_data, u_data, x_eqns, t_eqns, x_pseudo, t_pseudo, u_pseudo):

        # Initial condition loss
        u_init_pred = self.net(x_init, t_init)[0]
        init_loss = mean_squared_error(u_init_pred, u_init)

        # Boundary condition loss (specific for diffusion-sorption)
        x_r_bound_grad = x_r_bound.clone().requires_grad_(True)
        u_l_bound_pred = self.net(x_l_bound, t_bound)[0]
        u_r_bound_pred = self.net(x_r_bound_grad, t_bound)[0]
        b = Boundary(x_r_bound_grad, u_r_bound_pred)

        bound_loss = mean_squared_error(u_l_bound_pred, 1.0) + mean_squared_error(u_r_bound_pred, b)

        # Data loss (weighted by 10)
        u_data_pred = self.net(x_data, t_data)[0]
        data_loss = mean_squared_error(u_data_pred, u_data)

        # PDE residual loss
        if x_eqns.shape[0] > 0:
            x_eqns_grad = x_eqns.clone().requires_grad_(True)
            t_eqns_grad = t_eqns.clone().requires_grad_(True)
            u_eqns_pred = self.net(x_eqns_grad, t_eqns_grad)[0]
            e = Diffusion_sorption(x_eqns_grad, t_eqns_grad, u_eqns_pred)
            eqns_loss = mean_squared_error(e, 0)
        else:
            eqns_loss = torch.tensor(0.0, device=self.device)

        # Pseudo-label loss
        if x_pseudo.shape[0] > 0:
            u_pseudo_pred = self.net(x_pseudo, t_pseudo)[0]
            pseudo_loss = mean_squared_error(u_pseudo_pred, u_pseudo)
        else:
            pseudo_loss = torch.tensor(0.0, device=self.device)

        # Total loss (note: data_loss weighted by 10)
        loss = init_loss + eqns_loss + 10 * data_loss + bound_loss + pseudo_loss

        return loss, init_loss, bound_loss, eqns_loss, data_loss, pseudo_loss

    def update_data_eqns_points(self):
        """Update pseudo-labeled and equation points"""
        self.net.eval()

        # Compute PDE residual for all sample points
        with torch.enable_grad():
            x_sample_grad = self.x_sample.clone().requires_grad_(True)
            t_sample_grad = self.t_sample.clone().requires_grad_(True)
            u_sample_pred = self.net(x_sample_grad, t_sample_grad)[0]
            e = Diffusion_sorption(x_sample_grad, t_sample_grad, u_sample_pred)

        e_np = torch.abs(e).detach().cpu().numpy()

        # Select points with lowest residual
        sample_size = self.t_sample.shape[0]
        pseudo_size = int(self.max_rate * sample_size)

        if pseudo_size == 0:
            idx_pseudo = []
        else:
            idx_pseudo = np.argpartition(e_np.T[0], pseudo_size)[:pseudo_size]

        # Update confidence flags
        self.flag_pseudo[idx_pseudo] += 1
        mask = np.ones(self.flag_pseudo.shape[0], dtype=np.bool_)
        mask[idx_pseudo] = False
        self.flag_pseudo[mask] = 0

        # Only use stable pseudo-labels
        idx_pseudo_stable = np.where(self.flag_pseudo > self.stab_coeff)[0]
        self.x_pseudo = self.x_sample[idx_pseudo_stable, :]
        self.t_pseudo = self.t_sample[idx_pseudo_stable, :]

        # Generate pseudo-labels
        if len(idx_pseudo_stable) > 0:
            with torch.no_grad():
                self.u_pseudo = self.net(self.x_pseudo, self.t_pseudo)[0].detach()
        else:
            self.u_pseudo = torch.empty(0, 1, device=self.device)

        # Remaining points are equation points
        mask = np.ones(self.flag_pseudo.shape[0], dtype=np.bool_)
        mask[idx_pseudo_stable] = False
        self.x_eqns = self.x_sample[mask]
        self.t_eqns = self.t_sample[mask]

        pseudo_count = self.t_pseudo.shape[0]
        eqns_count = self.t_eqns.shape[0]
        self.logging(f"Number Pseudo Points: {pseudo_count}, Number Eqns Points: {eqns_count}")

        self.net.train()

    def train(self, max_time, adam_it):
        """Train with self-training"""
        self.it = 0
        self.total_time = 0
        self.start_time = time.time()

        # Initialize pseudo-label split
        self.update_data_eqns_points()

        self.net.train()

        while self.it < adam_it and self.total_time < max_time:

            # Sample batches
            N_eqns = self.t_eqns.shape[0]
            if N_eqns > 0:
                idx_batch = np.random.choice(N_eqns, min(self.batch_size, N_eqns), replace=False)
                x_eqns_batch = self.x_eqns[idx_batch, :]
                t_eqns_batch = self.t_eqns[idx_batch, :]
            else:
                x_eqns_batch = torch.empty(0, 1, device=self.device)
                t_eqns_batch = torch.empty(0, 1, device=self.device)

            N_pseudo = self.t_pseudo.shape[0]
            if N_pseudo > 0:
                idx_batch = np.random.choice(N_pseudo, min(self.batch_size, N_pseudo), replace=False)
                x_pseudo_batch = self.x_pseudo[idx_batch, :]
                t_pseudo_batch = self.t_pseudo[idx_batch, :]
                u_pseudo_batch = self.u_pseudo[idx_batch, :]
            else:
                x_pseudo_batch = torch.empty(0, 1, device=self.device)
                t_pseudo_batch = torch.empty(0, 1, device=self.device)
                u_pseudo_batch = torch.empty(0, 1, device=self.device)

            # Training step
            self.optimizer.zero_grad()

            loss, init_loss, bound_loss, eqns_loss, data_loss, pseudo_loss = self.compute_loss(
                self.x_init, self.t_init, self.u_init,
                self.x_l_bound, self.x_r_bound, self.t_bound,
                self.x_data, self.t_data, self.u_data,
                x_eqns_batch, t_eqns_batch,
                x_pseudo_batch, t_pseudo_batch, u_pseudo_batch
            )

            loss.backward()
            self.optimizer.step()

            # Logging
            if self.it % 10 == 0:
                elapsed = time.time() - self.start_time
                self.total_time += elapsed / 3600.0
                log_item = 'It: %d, Loss: %.3e, Init Loss: %.3e, Bound Loss: %.3e, Eqns Loss: %.3e, ' \
                           'Data Loss: %.3e, Pseudo Loss: %.3e, Time: %.2fs, Total Time: %.2fh' % \
                           (self.it, loss.item(), init_loss.item(), bound_loss.item(),
                            eqns_loss.item(), data_loss.item(), pseudo_loss.item(), elapsed, self.total_time)
                self.logging(log_item)
                self.start_time = time.time()

            # Evaluation
            if self.it % 100 == 0:
                u_pred = self.predict(self.x_test, self.t_test)
                error_u = relative_error(u_pred, self.u_test.cpu().numpy())
                log_item = 'Error u: %e' % (error_u)
                self.logging(log_item)

            # Update pseudo-labeled points
            if self.it % self.update_freq == 0 and self.it > 0:
                self.update_data_eqns_points()

            self.it += 1

    def predict(self, x_star, t_star):
        """Make predictions"""
        self.net.eval()

        if isinstance(x_star, np.ndarray):
            x_star = torch.tensor(x_star, dtype=torch.float32, device=self.device)
            t_star = torch.tensor(t_star, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            u_star = self.net(x_star, t_star)[0]

        self.net.train()
        return u_star.cpu().numpy()

    def logging(self, log_item):
        """Log progress"""
        with open(self.log_path, 'a+') as log:
            log.write(log_item + '\n')
        print(log_item)


if __name__ == '__main__':
    xL, xR = 0, 1
    N_init = 5000
    N_bound = 1000
    N_data = 1000
    N_test = 20000
    batch_size = 20000
    layers = [2] + 4 * [32] + [1]
    create_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    log_path = "../output/log/diffsorb1D-stpinn-%s" % (create_date)

    # Self-training parameters
    update_freq = 100
    max_rate = 0.999
    stab_coeff = 1

    ### load data
    data_path = r'../input/diffsorb1D.npy'
    data = np.load(data_path, allow_pickle=True)
    x = data.item()['x']
    t = data.item()['t']
    u = data.item()['u']

    # init
    idx_init = np.where(t == 0.0)[0]
    x_init = x[idx_init, :]
    t_init = t[idx_init, :]
    u_init = u[idx_init, :]

    # boundary
    idx_bound = np.where(x == x[0, 0])[0]
    t_bound = t[idx_bound, :]
    x_l_bound = xL * np.ones_like(t_bound)
    x_r_bound = xR * np.ones_like(t_bound)

    ### rearrange data
    x_sample = x
    t_sample = t

    # initial
    idx_init = np.random.choice(x_init.shape[0], min(N_init, x_init.shape[0]), replace=False)
    x_init = x_init[idx_init, :]
    t_init = t_init[idx_init, :]
    u_init = u_init[idx_init, :]

    # boundary
    idx_bound = np.random.choice(t_bound.shape[0], min(N_bound, t_bound.shape[0]), replace=False)
    x_l_bound = x_l_bound[idx_bound, :]
    x_r_bound = x_r_bound[idx_bound, :]
    t_bound = t_bound[idx_bound, :]

    # intra-domain
    idx_data = np.random.choice(x.shape[0], min(N_data, x.shape[0]), replace=False)
    x_data = x[idx_data, :]
    t_data = t[idx_data, :]
    u_data = u[idx_data, :]

    # test
    idx_test = np.random.choice(x.shape[0], min(N_test, x.shape[0]), replace=False)
    x_test = x[idx_test, :]
    t_test = t[idx_test, :]
    u_test = u[idx_test, :]

    ## build model
    model = SelfTrainingPINN(x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_sample, t_sample, x_data, t_data,
                              u_data, x_test, t_test, u_test, batch_size, layers, log_path, update_freq, max_rate,
                              stab_coeff)

    ### train
    model.train(max_time=10, adam_it=20000)

    ### test
    u_pred = model.predict(x, t)
    error_u = relative_error(u_pred, u)
    model.logging('L2 error u: %e' % (error_u))

    u_pred = model.predict(x, t)
    error_u = mean_squared_error(u_pred, u)
    model.logging('MSE error u: %e' % (error_u))

    # save prediction
    data_output_path = "../output/prediction/diffsorb1D-stpinn-%s.npy" % (create_date)
    data_output = {'u': u_pred}
    np.save(data_output_path, data_output)