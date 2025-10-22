import time
import numpy as np
import torch
from source.pdes import diffusion_reaction_1d
from source.utilities import NeuralNet, mean_squared_error, relative_error, set_random_seed, get_device

set_random_seed(1234)


class PhysicsInformedNN:
    """Physics-Informed Neural Network for 1D Diffusion-Reaction Equation"""

    def __init__(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_eqns, t_eqns,
                 x_data, t_data, u_data, x_test, t_test, u_test, nu, rho, batch_size, layers, log_path, device=None):
        """
        Initialize the PINN model.

        Args:
            x_init, t_init, u_init: Initial condition points
            x_l_bound, x_r_bound, t_bound: Boundary condition points
            x_eqns, t_eqns: Collocation points for PDE
            x_data, t_data, u_data: Training data points
            x_test, t_test, u_test: Test data points
            nu: Diffusion coefficient
            rho: Reaction rate
            batch_size: Batch size for training
            layers: Network architecture
            log_path: Path for logging
            device: torch device (cuda/cpu)
        """
        # Device setup
        self.device = device if device is not None else get_device()

        # Convert all numpy arrays to torch tensors and move to device
        self.x_init = torch.tensor(x_init, dtype=torch.float32, device=self.device)
        self.t_init = torch.tensor(t_init, dtype=torch.float32, device=self.device)
        self.u_init = torch.tensor(u_init, dtype=torch.float32, device=self.device)

        self.x_l_bound = torch.tensor(x_l_bound, dtype=torch.float32, device=self.device)
        self.x_r_bound = torch.tensor(x_r_bound, dtype=torch.float32, device=self.device)
        self.t_bound = torch.tensor(t_bound, dtype=torch.float32, device=self.device)

        self.x_eqns = torch.tensor(x_eqns, dtype=torch.float32, device=self.device)
        self.t_eqns = torch.tensor(t_eqns, dtype=torch.float32, device=self.device)

        self.x_data = torch.tensor(x_data, dtype=torch.float32, device=self.device)
        self.t_data = torch.tensor(t_data, dtype=torch.float32, device=self.device)
        self.u_data = torch.tensor(u_data, dtype=torch.float32, device=self.device)

        self.x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)
        self.t_test = torch.tensor(t_test, dtype=torch.float32, device=self.device)
        self.u_test = torch.tensor(u_test, dtype=torch.float32, device=self.device)

        self.nu = nu
        self.rho = rho
        self.layers = layers
        self.log_path = log_path
        self.batch_size = batch_size

        # Initialize neural network
        self.net = NeuralNet(x_eqns, t_eqns, layers=self.layers, device=self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def compute_loss(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound,
                     x_data, t_data, u_data, x_eqns, t_eqns):
        """
        Compute total loss (initial + boundary + data + PDE residual).

        Args:
            x_init, t_init, u_init: Initial condition tensors
            x_l_bound, x_r_bound, t_bound: Boundary condition tensors
            x_data, t_data, u_data: Data tensors
            x_eqns, t_eqns: Collocation point tensors

        Returns:
            loss, init_loss, bound_loss, eqns_loss, data_loss
        """
        # Initial condition loss
        u_init_pred = self.net(x_init, t_init)[0]
        init_loss = mean_squared_error(u_init_pred, u_init)

        # Boundary condition loss (periodic)
        u_l_bound_pred = self.net(x_l_bound, t_bound)[0]
        u_r_bound_pred = self.net(x_r_bound, t_bound)[0]
        bound_loss = mean_squared_error(u_l_bound_pred, u_r_bound_pred)

        # Data loss
        u_data_pred = self.net(x_data, t_data)[0]
        data_loss = mean_squared_error(u_data_pred, u_data)

        # PDE residual loss (requires gradients)
        x_eqns_grad = x_eqns.clone().requires_grad_(True)
        t_eqns_grad = t_eqns.clone().requires_grad_(True)
        u_eqns_pred = self.net(x_eqns_grad, t_eqns_grad)[0]
        e = diffusion_reaction_1d(x_eqns_grad, t_eqns_grad, u_eqns_pred, self.nu, self.rho)
        eqns_loss = mean_squared_error(e, 0)

        # Total loss
        loss = init_loss + eqns_loss + data_loss + bound_loss

        return loss, init_loss, bound_loss, eqns_loss, data_loss

    def train(self, max_time, adam_it):
        """
        Train the model using Adam optimizer.

        Args:
            max_time: Maximum training time in hours
            adam_it: Maximum number of Adam iterations
        """
        N_eqns = self.t_eqns.shape[0]
        self.start_time = time.time()
        self.total_time = 0
        self.it = 0

        self.net.train()

        while self.it < adam_it and self.total_time < max_time:

            # Random batch selection
            idx_batch = np.random.choice(N_eqns, min(self.batch_size, N_eqns), replace=False)
            x_eqns_batch = self.x_eqns[idx_batch, :]
            t_eqns_batch = self.t_eqns[idx_batch, :]

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute loss
            loss, init_loss, bound_loss, eqns_loss, data_loss = self.compute_loss(
                self.x_init, self.t_init, self.u_init,
                self.x_l_bound, self.x_r_bound, self.t_bound,
                self.x_data, self.t_data, self.u_data,
                x_eqns_batch, t_eqns_batch
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Print progress
            if self.it % 10 == 0:
                elapsed = time.time() - self.start_time
                self.total_time += elapsed / 3600.0
                log_item = 'It: %d, Loss: %.3e, Init Loss: %.3e, Bound Loss: %.3e, Eqns Loss: %.3e, ' \
                           'Data Loss: %.3e, Time: %.2fs, Total Time: %.2fh' % \
                           (self.it, loss.item(), init_loss.item(), bound_loss.item(),
                            eqns_loss.item(), data_loss.item(), elapsed, self.total_time)
                self.logging(log_item)
                self.start_time = time.time()

            # Evaluate
            if self.it % 100 == 0:
                u_pred = self.predict(self.x_test, self.t_test)
                error_u = relative_error(u_pred, self.u_test.cpu().numpy())
                log_item = 'Error u: %e' % (error_u)
                self.logging(log_item)

            self.it += 1

    def predict(self, x_star, t_star):
        """
        Make predictions at given points.

        Args:
            x_star: x coordinates (numpy array or tensor)
            t_star: t coordinates (numpy array or tensor)

        Returns:
            u_star: Predicted solution (numpy array)
        """
        self.net.eval()

        # Convert to tensors if necessary
        if isinstance(x_star, np.ndarray):
            x_star = torch.tensor(x_star, dtype=torch.float32, device=self.device)
            t_star = torch.tensor(t_star, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            u_star = self.net(x_star, t_star)[0]

        self.net.train()

        return u_star.cpu().numpy()

    def logging(self, log_item):
        """Log training progress to file and console."""
        with open(self.log_path, 'a+') as log:
            log.write(log_item + '\n')
        print(log_item)


if __name__ == '__main__':
    xL, xR = 0, 1
    nu = 0.5
    rho = 1.0
    N_init = 5000
    N_bound = 1000
    N_data = 1000
    N_test = 20000
    batch_size = 20000
    layers = [2] + 4 * [32] + [1]
    create_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    log_path = "../output/log/diffreact1D-pinn-%s" % (create_date)

    ### load data
    data_path = r'../input/diffreact1D.npy'
    data = np.load(data_path, allow_pickle=True)
    x = data.item()['x']
    t = data.item()['t']
    u = data.item()['u']

    ### arrange data
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
    x_eqns = x
    t_eqns = t
    u_eqns = u

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

    model = PhysicsInformedNN(x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_eqns, t_eqns, x_data, t_data,
                              u_data, x_test, t_test, u_test, nu, rho, batch_size, layers, log_path)

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
    data_output_path = "../output/prediction/diffreact1d-pinn-%s.npy" % (create_date)
    data_output = {'u': u_pred}
    np.save(data_output_path, data_output)