"""
comparative_viz.py: Visualization utilities for PINNs and ST-PINNs for Diffusion-Reaction Equation

This library provides functions to visualize training metrics (loss, error) and analyze solution
quality for Physics-Informed Neural Networks (PINN) and Self-Training PINNs (ST-PINNs).

ASSUMPTION: Training scripts (DiffReact1D_PINN.py, DiffReact1D_STPINN.py) save structured history
(loss, error, prediction data) into `.npy` files.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata


def create_output_dirs(base_path):
    """
    Create output directories for plots and logs.

    Args:
        base_path: str. Base directory path.

    Returns:
        dict: Dictionary of subdirectory paths.
    """
    dirs = {
        'log': os.path.join(base_path, 'log'),  # <-- ADDED THIS LINE
        'plots': os.path.join(base_path, 'plots'),
        'loss': os.path.join(base_path, 'plots', 'loss'),
        'solution': os.path.join(base_path, 'plots', 'solution'),
        'error': os.path.join(base_path, 'plots', 'error'),
        'pseudo': os.path.join(base_path, 'plots', 'pseudo')  # For pseudo-label history
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def plot_loss_history(loss_history, save_path, equation_name=""):
    """
    Plot loss history for PINN or ST-PINN.

    Args:
        loss_history (dict): Dictionary containing keys 'iteration', 'total_loss', 'init_loss',
                             'bound_loss', 'data_loss', 'eqns_loss', and optionally 'pseudo_loss'.
        save_path (str): Path to save the plot.
        equation_name (str): Name of the equation for the title.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    iterations = loss_history['iteration']

    # Plot total loss
    axes[0].semilogy(iterations, loss_history['total_loss'], 'k-', linewidth=2, label='Total Loss')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{equation_name} - Training Loss History', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot individual loss components
    axes[1].semilogy(iterations, loss_history['init_loss'], label='Initial Condition', linewidth=1.5)
    axes[1].semilogy(iterations, loss_history['bound_loss'], label='Boundary Condition', linewidth=1.5)
    axes[1].semilogy(iterations, loss_history['eqns_loss'], label='PDE Residual', linewidth=1.5)
    axes[1].semilogy(iterations, loss_history['data_loss'], label='Data', linewidth=1.5)

    if 'pseudo_loss' in loss_history and len(loss_history['pseudo_loss']) > 0:
        axes[1].semilogy(iterations, loss_history['pseudo_loss'], label='Pseudo-Label Loss', linewidth=1.5)

    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Loss Components', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss history plot saved to: {save_path}")


def plot_error_history(error_history, save_path, equation_name=""):
    """
    Plot test error history for PINN or ST-PINN.

    Args:
        error_history (dict): Dictionary with keys 'iteration' and 'error'.
        save_path (str): Path to save the plot.
        equation_name (str): Name of the equation for the title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = error_history['iteration']
    errors = error_history['error']

    ax.semilogy(iterations, errors, 'b-', linewidth=2, marker='o', markersize=4, label='Relative L2 Error')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title(f'{equation_name} - Test Error History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error history plot saved to: {save_path}")


def plot_pseudo_label_history(pseudo_history, save_path, equation_name=""):
    """
    Plot pseudo-label history for ST-PINN.

    Args:
        pseudo_history (dict): Dictionary with keys 'iteration', 'count'.
        save_path (str): Path to save the plot.
        equation_name (str): Name of the equation for the title.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = pseudo_history['iteration']
    count = pseudo_history['count']

    ax.plot(iterations, count, 'g-', linewidth=2, marker='o', markersize=4, label='Pseudo-Labels Count')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{equation_name} - Pseudo-Label History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pseudo-label history plot saved to: {save_path}")


def plot_solution_comparison_1d(x, t, u_true, u_pred, save_path, equation_name=""):
    """
    Plot ground truth solution, predicted solution, and error.

    Args:
        x, t (numpy.ndarray): Spatial and temporal coordinates.
        u_true (numpy.ndarray): Ground truth solution.
        u_pred (numpy.ndarray): Predicted solution.
        save_path (str): Path to save the plot.
        equation_name (str): Equation name for the title.
    """
    # Flatten arrays
    x = x.flatten()
    t = t.flatten()
    u_true = u_true.flatten()
    u_pred = u_pred.flatten()

    x_unique = np.unique(x)
    t_unique = np.unique(t)

    # Check for sufficient unique points
    if len(x_unique) < 2 or len(t_unique) < 2:
        print("Insufficient unique points for 2D plot.")
        return

    X, T = np.meshgrid(x_unique, t_unique)

    # Interpolate solutions
    U_true = griddata((x, t), u_true, (X, T), method='linear')
    U_pred = griddata((x, t), u_pred, (X, T), method='linear')
    error = np.abs(U_true - U_pred)

    fig = plt.figure(figsize=(18, 5))
    gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    # Ground truth
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(U_true, aspect='auto', cmap='viridis', origin='lower',
                     extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()])
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1)

    # Prediction
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(U_pred, aspect='auto', cmap='viridis', origin='lower',
                     extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()])
    ax2.set_title('Prediction', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2)

    # Error
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(error, aspect='auto', cmap='hot', origin='lower',
                     extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()])
    ax3.set_title('Absolute Error', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=ax3)

    fig.suptitle(f"{equation_name} - Solution Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Solution comparison saved to: {save_path}")


def plot_solution_snapshots_1d(x, t, u_true, u_pred, time_snapshots, save_path, equation_name=""):
    """
    Plot solution snapshots at specific time points for PINN or ST-PINN.

    Args:
        x, t (numpy.ndarray): Spatial and temporal coordinates.
        u_true (numpy.ndarray): Ground truth solution.
        u_pred (numpy.ndarray): Predicted solution.
        time_snapshots (list): List of time values for snapshots.
        save_path (str): Path to save the plot.
        equation_name (str): Equation name for the title.
    """
    x = x.flatten()
    t = t.flatten()
    u_true = u_true.flatten()
    u_pred = u_pred.flatten()

    fig, axes = plt.subplots(1, len(time_snapshots), figsize=(5 * len(time_snapshots), 4))

    if len(time_snapshots) == 1:
        axes = [axes]  # Make axes iterable

    for i, t_snap in enumerate(time_snapshots):
        # Select points at this time
        idx = np.abs(t - t_snap) < 1e-6
        if not np.any(idx):
            print(f"Warning: No points found for t = {t_snap:.3f}")
            continue

        x_snap = x[idx]
        u_true_snap = u_true[idx]
        u_pred_snap = u_pred[idx]

        # Sort by spatial coordinates (x)
        sort_idx = np.argsort(x_snap)
        x_snap = x_snap[sort_idx]
        u_true_snap = u_true_snap[sort_idx]
        u_pred_snap = u_pred_snap[sort_idx]

        axes[i].plot(x_snap, u_true_snap, 'b-', linewidth=2, label='Ground Truth')
        axes[i].plot(x_snap, u_pred_snap, 'r--', linewidth=2, label='Prediction')
        axes[i].set_xlabel('x', fontsize=12)
        axes[i].set_ylabel('u', fontsize=12)
        axes[i].set_title(f't = {t_snap:.3f}', fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    if equation_name:
        fig.suptitle(f'{equation_name} - Solution Snapshots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Solution snapshots plot saved to: {save_path}")


def plot_error_distribution(u_true, u_pred, save_path, equation_name=""):
    """
    Plot error distribution (absolute and relative errors) as histograms.

    Args:
        u_true (numpy.ndarray): Ground truth solution.
        u_pred (numpy.ndarray): Predicted solution.
        save_path (str): Path to save the plot.
        equation_name (str): Name of the equation for the title.
    """
    u_true = u_true.flatten()
    u_pred = u_pred.flatten()

    abs_error = np.abs(u_true - u_pred)
    rel_error = abs_error / (np.abs(u_true) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute error histogram
    axes[0].hist(abs_error, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Absolute Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Absolute Error Distribution', fontsize=13, fontweight='bold')

    # Relative error histogram
    axes[1].hist(rel_error, bins=50, color='orange', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Relative Error', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Relative Error Distribution', fontsize=13, fontweight='bold')

    fig.suptitle(f'{equation_name}: Error Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Error distribution saved to: {save_path}")