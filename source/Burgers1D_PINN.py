import time
import numpy as np
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.pdes import Burgers1D
from source.utilities import NeuralNet, mean_squared_error, relative_error, set_random_seed, get_device
from source.visualization import (plot_loss_history, plot_error_history, plot_solution_comparison_1d,
                                   plot_solution_snapshots_1d, plot_error_distribution, create_output_dirs)

set_random_seed(1234)


class PhysicsInformedNN:
    """Physics-Informed Neural Network for 1D Burgers Equation"""

    def __init__(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_eqns, t_eqns, u_eqns,
                 x_data, t_data, u_data, x_test, t_test, u_test, nu, batch_size, layers, log_path, device=None):
        """
        Initialize the PINN model.

        Args:
            x_init, t_init, u_init: Initial condition points
            x_l_bound, x_r_bound, t_bound: Boundary condition points
            x_eqns, t_eqns, u_eqns: Collocation points for PDE
            x_data, t_data, u_data: Training data points
            x_test, t_test, u_test: Test data points
            nu: Viscosity parameter
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
        self.layers = layers
        self.log_path = log_path
        self.batch_size = batch_size

        # Initialize neural network
        self.net = NeuralNet(x_eqns, t_eqns, layers=self.layers, device=self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        # History tracking for visualization
        self.loss_history = {
            'iteration': [],
            'total_loss': [],
            'init_loss': [],
            'bound_loss': [],
            'eqns_loss': [],
            'data_loss': []
        }
        self.error_history = {
            'iteration': [],
            'error': []
        }

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
        e = Burgers1D(x_eqns_grad, t_eqns_grad, u_eqns_pred, self.nu)
        eqns_loss = mean_squared_error(e, 0)

        # Total loss
        loss = init_loss + eqns_loss + bound_loss

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

            # Print progress and record loss history
            if self.it % 10 == 0:
                elapsed = time.time() - self.start_time
                self.total_time += elapsed / 3600.0
                log_item = 'It: %d, Loss: %.3e, Init Loss: %.3e, Bound Loss: %.3e, Eqns Loss: %.3e, ' \
                           'Data Loss: %.3e, Time: %.2fs, Total Time: %.2fh' % \
                           (self.it, loss.item(), init_loss.item(), bound_loss.item(),
                            eqns_loss.item(), data_loss.item(), elapsed, self.total_time)
                self.logging(log_item)
                self.start_time = time.time()

                # Record loss history
                self.loss_history['iteration'].append(self.it)
                self.loss_history['total_loss'].append(loss.item())
                self.loss_history['init_loss'].append(init_loss.item())
                self.loss_history['bound_loss'].append(bound_loss.item())
                self.loss_history['eqns_loss'].append(eqns_loss.item())
                self.loss_history['data_loss'].append(data_loss.item())

            # Evaluate and record error history
            if self.it % 100 == 0:
                u_pred = self.predict(self.x_test, self.t_test)
                error_u = relative_error(u_pred, self.u_test.cpu().numpy())
                log_item = 'Error u: %e' % (error_u)
                self.logging(log_item)

                # Record error history
                self.error_history['iteration'].append(self.it)
                self.error_history['error'].append(error_u)

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
    # Get the project root directory (parent of source/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    xL, xR = 0.0, 1.0
    nu = 0.01
    N_init = 5000
    N_bound = 1000
    N_data = 1000
    N_test = 20000
    batch_size = 20000
    layers = [2] + 4 * [32] + [1]
    create_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))

    # Create output directories
    os.makedirs(os.path.join(project_root, "output", "log"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "output", "prediction"), exist_ok=True)

    log_path = os.path.join(project_root, "output", "log", f"burgers1D-pinn-{create_date}")

    ### load data
    data_path = os.path.join(project_root, "input", "burgers1D.npy")
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

    model = PhysicsInformedNN(x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_eqns, t_eqns, u_eqns,
                              x_data, t_data, u_data, x_test, t_test, u_test, nu, batch_size, layers, log_path)

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
    data_output_path = os.path.join(project_root, "output", "prediction", f"burgers1d-pinn-{create_date}.npy")
    data_output = {'u': u_pred}
    np.save(data_output_path, data_output)

    ### Generate visualizations
    model.logging("\nGenerating visualizations...")
    output_dirs = create_output_dirs(os.path.join(project_root, "output"))

    # Plot loss history
    loss_plot_path = f"{output_dirs['loss']}/burgers1d-pinn-loss-{create_date}.png"
    plot_loss_history(model.loss_history, loss_plot_path)

    # Plot error history
    error_plot_path = f"{output_dirs['error']}/burgers1d-pinn-error-{create_date}.png"
    plot_error_history(model.error_history, error_plot_path)

    # Plot solution comparison
    solution_plot_path = f"{output_dirs['solution']}/burgers1d-pinn-solution-{create_date}.png"
    plot_solution_comparison_1d(x, t, u, u_pred, solution_plot_path, equation_name="Burgers 1D (PINN)")

    # Plot solution snapshots at different times
    t_unique = np.unique(t)
    time_snapshots = [t_unique[int(i * len(t_unique) / 4)] for i in range(4)]
    snapshot_plot_path = f"{output_dirs['solution']}/burgers1d-pinn-snapshots-{create_date}.png"
    plot_solution_snapshots_1d(x, t, u, u_pred, time_snapshots, snapshot_plot_path, equation_name="Burgers 1D (PINN)")

    # Plot error distribution
    error_dist_path = f"{output_dirs['error']}/burgers1d-pinn-error-dist-{create_date}.png"
    plot_error_distribution(u, u_pred, error_dist_path, equation_name="Burgers 1D (PINN)")

    model.logging("All visualizations saved successfully!")