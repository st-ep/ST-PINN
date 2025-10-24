import time
import numpy as np
import torch
import os
from source.pdes import diffusion_reaction_1d
from source.utilities import NeuralNet, mean_squared_error, relative_error, set_random_seed, get_device
from source.DiffReact_visualization import (plot_loss_history, plot_error_history, plot_solution_comparison_1d,
                                            plot_solution_snapshots_1d, plot_error_distribution,
                                            plot_pseudo_label_history,
                                            create_output_dirs)

set_random_seed(1234)


class SelfTrainingPINN:
    """Self-Training Physics-Informed Neural Network for 1D Diffusion-Reaction Equation"""

    def __init__(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_sample, t_sample, x_data, t_data,
                 u_data, x_test, t_test, u_test, nu, rho, batch_size, layers, log_path, update_freq, max_rate,
                 stab_coeff, device=None):

        # Device setup
        self.device = device if device is not None else get_device()

        # Convert data to torch tensors (Note: x_sample and t_sample are the original numpy inputs here)
        self.x_init = torch.tensor(x_init, dtype=torch.float32, device=self.device)
        self.t_init = torch.tensor(t_init, dtype=torch.float32, device=self.device)
        self.u_init = torch.tensor(u_init, dtype=torch.float32, device=self.device)

        self.x_l_bound = torch.tensor(x_l_bound, dtype=torch.float32, device=self.device)
        self.x_r_bound = torch.tensor(x_r_bound, dtype=torch.float32, device=self.device)
        self.t_bound = torch.tensor(t_bound, dtype=torch.float32, device=self.device)

        # ST-PINN specific: Sample points and currently selected pseudo points
        self.x_sample = torch.tensor(x_sample, dtype=torch.float32, device=self.device)
        self.t_sample = torch.tensor(t_sample, dtype=torch.float32, device=self.device)
        self.x_pseudo = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        self.t_pseudo = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        self.u_pseudo = torch.empty((0, 1), dtype=torch.float32, device=self.device)

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
        self.update_freq = update_freq
        self.max_rate = max_rate
        self.stab_coeff = stab_coeff

        # History tracking for visualization
        self.loss_history = {
            'iteration': [],
            'total_loss': [],
            'init_loss': [],
            'bound_loss': [],
            'eqns_loss': [],
            'data_loss': [],
            'pseudo_loss': [],  # ST-PINN specific
        }
        self.error_history = {
            'iteration': [],
            'error': [],
        }
        self.pseudo_history = {  # ST-PINN specific
            'iteration': [],
            'count': [],
        }

        # Initialize neural network
        self.net = NeuralNet(x_sample, t_sample, layers=self.layers, device=self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

    def update_pseudo_labels(self):
        """
        Calculates residual error and selects low-residual points as pseudo-labeled data.
        """
        self.net.eval()
        x_sample_grad = self.x_sample.clone().requires_grad_(True)
        t_sample_grad = self.t_sample.clone().requires_grad_(True)
        u_sample_pred = self.net(x_sample_grad, t_sample_grad)[0]

        # Calculate residual
        e = diffusion_reaction_1d(x_sample_grad, t_sample_grad, u_sample_pred, self.nu, self.rho)
        residual = torch.abs(e).detach().cpu().numpy().flatten()

        # Calculate max residual (stability)
        max_residual = np.max(residual)

        # Select points below the stability coefficient * max residual
        threshold = self.stab_coeff * max_residual
        idx_pseudo = np.where(residual < threshold)[0]

        # Limit the rate of growth for the pseudo-set size
        N_max_pseudo = int(len(self.x_sample) * self.max_rate)
        if len(idx_pseudo) > N_max_pseudo:
            idx_pseudo = np.random.choice(idx_pseudo, N_max_pseudo, replace=False)

        # Assign pseudo labels
        self.x_pseudo = self.x_sample[idx_pseudo, :]
        self.t_pseudo = self.t_sample[idx_pseudo, :]

        # Calculate current predictions for the pseudo labels
        with torch.no_grad():
            self.u_pseudo = self.net(self.x_pseudo, self.t_pseudo)[0]

        self.net.train()
        return len(self.x_pseudo)

    def compute_loss(self, x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_data, t_data, u_data,
                     x_sample_batch, t_sample_batch):
        """
        Compute total loss (initial + boundary + data + PDE residual + pseudo-label).
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

        # PDE residual loss
        x_sample_grad = x_sample_batch.clone().requires_grad_(True)
        t_sample_grad = t_sample_batch.clone().requires_grad_(True)
        u_sample_pred = self.net(x_sample_grad, t_sample_grad)[0]
        e = diffusion_reaction_1d(x_sample_grad, t_sample_grad, u_sample_pred, self.nu, self.rho)
        eqns_loss = mean_squared_error(e, 0)

        # Pseudo-label loss
        if len(self.x_pseudo) > 0:
            u_pseudo_pred = self.net(self.x_pseudo, self.t_pseudo)[0]
            pseudo_loss = mean_squared_error(u_pseudo_pred, self.u_pseudo.detach())
        else:
            pseudo_loss = torch.tensor(0.0, device=self.device)

        # Total loss
        loss = init_loss + eqns_loss + data_loss + bound_loss + pseudo_loss

        return loss, init_loss, bound_loss, eqns_loss, data_loss, pseudo_loss

    def train(self, max_time, adam_it):
        """
        Train the model using Adam optimizer with self-training updates.
        """
        N_sample = self.t_sample.shape[0]
        self.start_time = time.time()
        self.total_time = 0
        self.it = 0
        current_pseudo_count = 0

        self.net.train()

        while self.it < adam_it and self.total_time < max_time:
            # Update pseudo labels periodically
            if self.it % self.update_freq == 0:
                current_pseudo_count = self.update_pseudo_labels()

            # Random batch selection from sample points
            idx_batch = np.random.choice(N_sample, min(self.batch_size, N_sample), replace=False)
            x_sample_batch = self.x_sample[idx_batch, :]
            t_sample_batch = self.t_sample[idx_batch, :]

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute loss
            loss, init_loss, bound_loss, eqns_loss, data_loss, pseudo_loss = self.compute_loss(
                self.x_init, self.t_init, self.u_init,
                self.x_l_bound, self.x_r_bound, self.t_bound,
                self.x_data, self.t_data, self.u_data,
                x_sample_batch, t_sample_batch
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Record loss and pseudo history
            if self.it % 10 == 0:
                elapsed = time.time() - self.start_time
                self.total_time += elapsed / 3600.0

                self.loss_history['iteration'].append(self.it)
                self.loss_history['total_loss'].append(loss.item())
                self.loss_history['init_loss'].append(init_loss.item())
                self.loss_history['bound_loss'].append(bound_loss.item())
                self.loss_history['eqns_loss'].append(eqns_loss.item())
                self.loss_history['data_loss'].append(data_loss.item())
                self.loss_history['pseudo_loss'].append(pseudo_loss.item())

                self.pseudo_history['iteration'].append(self.it)
                self.pseudo_history['count'].append(current_pseudo_count)

                log_item = f'It: {self.it}, Loss: {loss.item():.3e}, Init: {init_loss.item():.3e}, Bound: {bound_loss.item():.3e}, ' \
                           f'Eqns: {eqns_loss.item():.3e}, Data: {data_loss.item():.3e}, Pseudo: {pseudo_loss.item():.3e}, ' \
                           f'Pseudo #: {current_pseudo_count}, Time: {elapsed:.2f}s, Total Time: {self.total_time:.2f}h'
                self.logging(log_item)
                self.start_time = time.time()

            # Evaluate and record error history
            if self.it % 100 == 0:
                u_pred = self.predict(self.x_test, self.t_test)
                error_u = relative_error(u_pred, self.u_test.cpu().numpy())
                self.error_history['iteration'].append(self.it)
                self.error_history['error'].append(error_u)

                log_item = f'Error u: {error_u:.3e}'
                self.logging(log_item)

            self.it += 1

    def predict(self, x_star, t_star):
        """
        Make predictions at given points.
        """
        self.net.eval()

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
    N_sample = 20000  # Total sample points for self-training
    N_test = 20000
    batch_size = 20000
    layers = [2] + 4 * [32] + [1]
    update_freq = 50  # Iterations between pseudo-label updates
    max_rate = 0.5  # Maximum percentage of points that can be pseudo-labeled
    stab_coeff = 0.1  # Stability coefficient for selecting pseudo points

    print("Main script working directory:", os.getcwd())

    # --- START MODIFICATION ---
    # Define the ABSOLUTE BASE PATH for all outputs
    BASE_OUTPUT_DIR = "/home/dhoussou/Documents/ST-PINN_Output"

    current_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    create_date = f"DR1D_STPINN_{current_time_str}"

    # Update log_path to use the absolute path
    log_path = os.path.join(BASE_OUTPUT_DIR, "log", f"diffreact1D-stpinn-{create_date}.log")

    # Create output directories for visualizations (including the 'log' directory)
    # NOTE: create_output_dirs must be modified to include 'log' creation (see DiffReact_visualization.py)
    output_dirs = create_output_dirs(BASE_OUTPUT_DIR)
    # --- END MODIFICATION ---

    ### Load data
    data_path = r'../input/diffreact1D.npy'
    data = np.load(data_path, allow_pickle=True)
    x = data.item()['x']
    t = data.item()['t']
    u = data.item()['u']

    ### Arrange data
    idx_init = np.where(t == 0.0)[0]
    x_init = x[idx_init, :]
    t_init = t[idx_init, :]
    u_init = u[idx_init, :]

    idx_bound = np.where(x == x[0, 0])[0]
    t_bound = t[idx_bound, :]
    x_l_bound = xL * np.ones_like(t_bound)
    x_r_bound = xR * np.ones_like(t_bound)

    x_sample = x
    t_sample = t

    idx_init = np.random.choice(x_init.shape[0], min(N_init, x_init.shape[0]), replace=False)
    x_init = x_init[idx_init, :]
    t_init = t_init[idx_init, :]
    u_init = u_init[idx_init, :]

    idx_bound = np.random.choice(t_bound.shape[0], min(N_bound, t_bound.shape[0]), replace=False)
    x_l_bound = x_l_bound[idx_bound, :]
    x_r_bound = x_r_bound[idx_bound, :]
    t_bound = t_bound[idx_bound, :]

    idx_data = np.random.choice(x.shape[0], min(N_data, x.shape[0]), replace=False)
    x_data = x[idx_data, :]
    t_data = t[idx_data, :]
    u_data = u[idx_data, :]

    idx_test = np.random.choice(x.shape[0], min(N_test, x.shape[0]), replace=False)
    x_test = x[idx_test, :]
    t_test = t[idx_test, :]
    u_test = u[idx_test, :]

    model = SelfTrainingPINN(x_init, t_init, u_init, x_l_bound, x_r_bound, t_bound, x_sample, t_sample, x_data, t_data,
                             u_data, x_test, t_test, u_test, nu, rho, batch_size, layers, log_path, update_freq,
                             max_rate, stab_coeff)

    ### Train model
    model.train(max_time=10, adam_it=20000)

    ### Test and final prediction
    u_pred = model.predict(x, t)
    error_u_l2 = relative_error(u_pred, u)
    model.logging(f'L2 error u: {error_u_l2:.3e}')

    error_u_mse = mean_squared_error(u_pred, u)
    model.logging(f'MSE error u: {error_u_mse:.3e}')

    ### Generate visualizations
    model.logging("\nGenerating visualizations...")

    # Plot loss history
    loss_plot_path = f"{output_dirs['loss']}/diffreact1D-stpinn-loss-{create_date}.png"
    plot_loss_history(model.loss_history, loss_plot_path, equation_name="Diffusion-Reaction 1D (ST-PINN)")

    # Plot error history
    error_plot_path = f"{output_dirs['error']}/diffreact1D-stpinn-error-{create_date}.png"
    plot_error_history(model.error_history, error_plot_path, equation_name="Diffusion-Reaction 1D (ST-PINN)")

    # Plot pseudo-label history
    pseudo_plot_path = f"{output_dirs['pseudo']}/diffreact1D-stpinn-pseudo-label-history-{create_date}.png"
    plot_pseudo_label_history(model.pseudo_history, pseudo_plot_path, equation_name="Diffusion-Reaction 1D (ST-PINN)")

    # Plot solution comparison
    solution_plot_path = f"{output_dirs['solution']}/diffreact1D-stpinn-solution-{create_date}.png"
    plot_solution_comparison_1d(x, t, u, u_pred, solution_plot_path, equation_name="Diffusion-Reaction 1D (ST-PINN)")

    # Plot solution snapshots
    t_unique = np.unique(t)
    time_snapshots = [t_unique[int(i * len(t_unique) / 4)] for i in range(4)]
    snapshot_plot_path = f"{output_dirs['solution']}/diffreact1D-stpinn-snapshots-{create_date}.png"
    plot_solution_snapshots_1d(x, t, u, u_pred, time_snapshots, snapshot_plot_path,
                               equation_name="Diffusion-Reaction 1D (ST-PINN)")

    # Plot error distribution
    error_dist_path = f"{output_dirs['error']}/diffreact1D-stpinn-error-dist-{create_date}.png"
    plot_error_distribution(u, u_pred, error_dist_path, equation_name="Diffusion-Reaction 1D (ST-PINN)")

    model.logging("All visualizations saved successfully!")