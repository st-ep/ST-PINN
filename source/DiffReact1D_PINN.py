import time
import numpy as np
import torch
import os  # Added for file saving
from source.pdes import diffusion_reaction_1d
from source.utilities import NeuralNet, mean_squared_error, relative_error, set_random_seed, get_device
# NOTE: Removed plot_pseudo_label_history since this is a standard PINN
from source.DiffReact_visualization import (plot_loss_history, plot_error_history, plot_solution_comparison_1d,
                                            plot_solution_snapshots_1d, plot_error_distribution, create_output_dirs)

set_random_seed(1234)


class PhysicsInformedNN:
    """Physics-Informed Neural Network for 1D Diffusion-Reaction Equation"""

    def __init__(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_eqns, t_eqns,
                 x_data, t_data, u_data, x_test, t_test, u_test, nu, rho, batch_size, layers, log_path, device=None):
        """
        Initialize the PINN model.
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

        # Initialize history tracking for visualization
        self.loss_history = {
            'iteration': [],
            'total_loss': [],
            'init_loss': [],
            'bound_loss': [],
            'eqns_loss': [],
            'data_loss': [],
        }
        self.error_history = {
            'iteration': [],
            'error': [],
        }

        # Initialize neural network
        self.net = NeuralNet(x_eqns, t_eqns, layers=self.layers, device=self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def compute_loss(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound,
                     x_data, t_data, u_data, x_eqns, t_eqns):
        """
        Compute total loss (initial + boundary + data + PDE residual).
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

            # Print progress and log loss history
            if self.it % 10 == 0:
                elapsed = time.time() - self.start_time
                self.total_time += elapsed / 3600.0

                # Record loss history
                self.loss_history['iteration'].append(self.it)
                self.loss_history['total_loss'].append(loss.item())
                self.loss_history['init_loss'].append(init_loss.item())
                self.loss_history['bound_loss'].append(bound_loss.item())
                self.loss_history['eqns_loss'].append(eqns_loss.item())
                self.loss_history['data_loss'].append(data_loss.item())

                log_item = 'It: %d, Loss: %.3e, Init Loss: %.3e, Bound Loss: %.3e, Eqns Loss: %.3e, ' \
                           'Data Loss: %.3e, Time: %.2fs, Total Time: %.2fh' % \
                           (self.it, loss.item(), init_loss.item(), bound_loss.item(),
                            eqns_loss.item(), data_loss.item(), elapsed, self.total_time)
                self.logging(log_item)
                self.start_time = time.time()

            # Evaluate and record error history
            if self.it % 100 == 0:
                u_pred = self.predict(self.x_test, self.t_test)
                error_u = relative_error(u_pred, self.u_test.cpu().numpy())

                # Record error history
                self.error_history['iteration'].append(self.it)
                self.error_history['error'].append(error_u)

                log_item = 'Error u: %e' % (error_u)
                self.logging(log_item)

            self.it += 1

    def predict(self, x_star, t_star):
        """
        Make predictions at given points.
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

    # --- START MODIFICATION ---
    # Define the ABSOLUTE BASE PATH for all outputs
    BASE_OUTPUT_DIR = "/home/dhoussou/Documents/PINN_Output"

    # Use a unique date/time for logging and data saving
    current_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    create_date = f"DR1D_PINN_{current_time_str}"

    # Update log_path to use the absolute path
    log_path = os.path.join(BASE_OUTPUT_DIR, "log", f"diffreact1D-pinn-{create_date}.log")

    # Create output directories for visualizations (including the 'log' directory, assuming DiffReact_visualization.py is updated)
    output_dirs = create_output_dirs(BASE_OUTPUT_DIR)
    # --- END MODIFICATION ---

    ### Load data
    data_path = r'../input/diffreact1D.npy'
    data = np.load(data_path, allow_pickle=True)
    x = data.item()['x']
    t = data.item()['t']
    u = data.item()['u']

    ### Arrange data
    # Initial condition
    idx_init = np.where(t == 0.0)[0]
    x_init = x[idx_init, :]
    t_init = t[idx_init, :]
    u_init = u[idx_init, :]

    # Boundary condition
    idx_bound = np.where(x == x[0, 0])[0]
    t_bound = t[idx_bound, :]
    x_l_bound = xL * np.ones_like(t_bound)
    x_r_bound = xR * np.ones_like(t_bound)

    # Rearrange data for training/testing
    x_eqns = x
    t_eqns = t
    u_eqns = u

    # Subsample data points for initial condition
    idx_init = np.random.choice(x_init.shape[0], min(N_init, x_init.shape[0]), replace=False)
    x_init = x_init[idx_init, :]
    t_init = t_init[idx_init, :]
    u_init = u_init[idx_init, :]

    # Subsample boundary condition points
    idx_bound = np.random.choice(t_bound.shape[0], min(N_bound, t_bound.shape[0]), replace=False)
    x_l_bound = x_l_bound[idx_bound, :]
    x_r_bound = x_r_bound[idx_bound, :]
    t_bound = t_bound[idx_bound, :]

    # Subsample intra-domain data points
    idx_data = np.random.choice(x.shape[0], min(N_data, x.shape[0]), replace=False)
    x_data = x[idx_data, :]
    t_data = t[idx_data, :]
    u_data = u[idx_data, :]

    # Subsample test points
    idx_test = np.random.choice(x.shape[0], min(N_test, x.shape[0]), replace=False)
    x_test = x[idx_test, :]
    t_test = t[idx_test, :]
    u_test = u[idx_test, :]

    model = PhysicsInformedNN(x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_eqns, t_eqns, x_data, t_data,
                              u_data, x_test, t_test, u_test, nu, rho, batch_size, layers, log_path)

    ### Train
    model.train(max_time=10, adam_it=20000)

    ### Test and final prediction
    u_pred = model.predict(x, t)
    error_u_l2 = relative_error(u_pred, u)
    model.logging(f'L2 error u: {error_u_l2:e}')

    error_u_mse = mean_squared_error(u_pred, u)
    model.logging(f'MSE error u: {error_u_mse:e}')

    ### Generate Visualizations
    model.logging("\nGenerating visualizations...")

    # Plot loss history
    loss_plot_path = f"{output_dirs['loss']}/diffreact1D-pinn-loss-{create_date}.png"
    plot_loss_history(model.loss_history, loss_plot_path, equation_name="Diffusion-Reaction 1D (PINN)")

    # Plot error history
    error_plot_path = f"{output_dirs['error']}/diffreact1D-pinn-error-{create_date}.png"
    plot_error_history(model.error_history, error_plot_path, equation_name="Diffusion-Reaction 1D (PINN)")

    # Plot solution comparison
    solution_plot_path = f"{output_dirs['solution']}/diffreact1D-pinn-solution-{create_date}.png"
    plot_solution_comparison_1d(x, t, u, u_pred, solution_plot_path, equation_name="Diffusion-Reaction 1D (PINN)")

    # Plot solution snapshots at specific time points
    t_unique = np.unique(t)
    time_snapshots = [t_unique[int(i * len(t_unique) / 4)] for i in range(4)]
    snapshot_plot_path = f"{output_dirs['solution']}/diffreact1D-pinn-snapshots-{create_date}.png"
    plot_solution_snapshots_1d(x, t, u, u_pred, time_snapshots, snapshot_plot_path,
                               equation_name="Diffusion-Reaction 1D (PINN)")

    # Plot error distribution
    error_dist_path = f"{output_dirs['error']}/diffreact1D-pinn-error-dist-{create_date}.png"
    plot_error_distribution(u, u_pred, error_dist_path, equation_name="Diffusion-Reaction 1D (PINN)")

    model.logging("All visualizations saved successfully!")