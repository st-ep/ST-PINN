"""
Visualization utilities for Physics-Informed Neural Networks
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


def plot_loss_history(loss_history, save_path):
    """
    Plot training loss history

    Args:
        loss_history: dict with keys 'iteration', 'total_loss', 'init_loss',
                      'bound_loss', 'eqns_loss', 'data_loss', and optionally 'pseudo_loss'
        save_path: path to save the figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    iterations = loss_history['iteration']

    # Plot total loss
    axes[0].semilogy(iterations, loss_history['total_loss'], 'k-', linewidth=2, label='Total Loss')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss History', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot individual loss components
    axes[1].semilogy(iterations, loss_history['init_loss'], label='Initial Condition', linewidth=1.5)
    axes[1].semilogy(iterations, loss_history['bound_loss'], label='Boundary Condition', linewidth=1.5)
    axes[1].semilogy(iterations, loss_history['eqns_loss'], label='PDE Residual', linewidth=1.5)
    axes[1].semilogy(iterations, loss_history['data_loss'], label='Data', linewidth=1.5)

    if 'pseudo_loss' in loss_history and len(loss_history['pseudo_loss']) > 0:
        axes[1].semilogy(iterations, loss_history['pseudo_loss'], label='Pseudo-Label', linewidth=1.5)

    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Loss Components', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss history plot saved to: {save_path}")


def plot_error_history(error_history, save_path):
    """
    Plot test error history

    Args:
        error_history: dict with keys 'iteration' and 'error'
        save_path: path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = error_history['iteration']
    errors = error_history['error']

    ax.semilogy(iterations, errors, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Test Error History', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error history plot saved to: {save_path}")


def plot_pseudo_label_history(pseudo_history, save_path):
    """
    Plot pseudo-label count history (for ST-PINN)

    Args:
        pseudo_history: dict with keys 'iteration', 'pseudo_count', 'eqns_count'
        save_path: path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = pseudo_history['iteration']

    ax.plot(iterations, pseudo_history['pseudo_count'], 'g-', linewidth=2,
            marker='s', markersize=4, label='Pseudo-Label Points')
    ax.plot(iterations, pseudo_history['eqns_count'], 'r-', linewidth=2,
            marker='^', markersize=4, label='Equation Points')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Number of Points', fontsize=12)
    ax.set_title('Point Distribution Evolution (Self-Training)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pseudo-label history plot saved to: {save_path}")


def plot_solution_comparison_1d(x, t, u_true, u_pred, save_path, equation_name=""):
    """
    Plot 1D solution comparison: prediction vs ground truth

    Args:
        x: spatial coordinates (N, 1)
        t: temporal coordinates (N, 1)
        u_true: true solution (N, 1)
        u_pred: predicted solution (N, 1)
        save_path: path to save the figure
        equation_name: name of the equation for title
    """
    # Flatten arrays
    x = x.flatten()
    t = t.flatten()
    u_true = u_true.flatten()
    u_pred = u_pred.flatten()

    # Create grid for plotting
    x_unique = np.unique(x)
    t_unique = np.unique(t)

    if len(x_unique) < 2 or len(t_unique) < 2:
        print("Warning: Not enough unique points for 2D plot")
        return

    X, T = np.meshgrid(x_unique, t_unique)

    # Interpolate solutions onto grid
    from scipy.interpolate import griddata
    U_true = griddata((x, t), u_true, (X, T), method='linear')
    U_pred = griddata((x, t), u_pred, (X, T), method='linear')

    # Compute error
    error = np.abs(U_true - U_pred)

    # Create figure
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    # Ground truth
    ax1 = plt.subplot(gs[0])
    im1 = ax1.imshow(U_true, aspect='auto', cmap='viridis',
                     extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
                     origin='lower', interpolation='bilinear')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('t', fontsize=12)
    ax1.set_title('Ground Truth', fontsize=13, fontweight='bold')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im1, cax=cax1)

    # Prediction
    ax2 = plt.subplot(gs[1])
    im2 = ax2.imshow(U_pred, aspect='auto', cmap='viridis',
                     extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
                     origin='lower', interpolation='bilinear')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('t', fontsize=12)
    ax2.set_title('Prediction', fontsize=13, fontweight='bold')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im2, cax=cax2)

    # Absolute error
    ax3 = plt.subplot(gs[2])
    im3 = ax3.imshow(error, aspect='auto', cmap='hot',
                     extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
                     origin='lower', interpolation='bilinear')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('t', fontsize=12)
    ax3.set_title('Absolute Error', fontsize=13, fontweight='bold')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im3, cax=cax3)

    if equation_name:
        fig.suptitle(f'{equation_name} - Solution Comparison', fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Solution comparison plot saved to: {save_path}")


def plot_solution_snapshots_1d(x, t, u_true, u_pred, time_snapshots, save_path, equation_name=""):
    """
    Plot solution snapshots at different time points

    Args:
        x: spatial coordinates (N, 1)
        t: temporal coordinates (N, 1)
        u_true: true solution (N, 1)
        u_pred: predicted solution (N, 1)
        time_snapshots: list of time values to plot
        save_path: path to save the figure
        equation_name: name of the equation for title
    """
    # Flatten arrays
    x = x.flatten()
    t = t.flatten()
    u_true = u_true.flatten()
    u_pred = u_pred.flatten()

    n_snapshots = len(time_snapshots)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(5*n_snapshots, 4))

    if n_snapshots == 1:
        axes = [axes]

    for i, t_val in enumerate(time_snapshots):
        # Find points closest to this time
        idx = np.abs(t - t_val) < 1e-6
        if not np.any(idx):
            # Find closest time
            t_closest = np.unique(t)[np.argmin(np.abs(np.unique(t) - t_val))]
            idx = np.abs(t - t_closest) < 1e-6
            t_val = t_closest

        x_snap = x[idx]
        u_true_snap = u_true[idx]
        u_pred_snap = u_pred[idx]

        # Sort by x
        sort_idx = np.argsort(x_snap)
        x_snap = x_snap[sort_idx]
        u_true_snap = u_true_snap[sort_idx]
        u_pred_snap = u_pred_snap[sort_idx]

        axes[i].plot(x_snap, u_true_snap, 'b-', linewidth=2, label='Ground Truth')
        axes[i].plot(x_snap, u_pred_snap, 'r--', linewidth=2, label='Prediction')
        axes[i].set_xlabel('x', fontsize=12)
        axes[i].set_ylabel('u', fontsize=12)
        axes[i].set_title(f't = {t_val:.3f}', fontsize=12, fontweight='bold')
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    if equation_name:
        fig.suptitle(f'{equation_name} - Solution Snapshots', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Solution snapshots plot saved to: {save_path}")


def plot_error_distribution(u_true, u_pred, save_path, equation_name=""):
    """
    Plot error distribution histogram

    Args:
        u_true: true solution (N, 1)
        u_pred: predicted solution (N, 1)
        save_path: path to save the figure
        equation_name: name of the equation for title
    """
    # Flatten arrays
    u_true = u_true.flatten()
    u_pred = u_pred.flatten()

    # Compute errors
    abs_error = np.abs(u_true - u_pred)
    rel_error = abs_error / (np.abs(u_true) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute error histogram
    axes[0].hist(abs_error, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Absolute Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Absolute Error Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axvline(np.mean(abs_error), color='r', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(abs_error):.2e}')
    axes[0].legend(fontsize=10)

    # Relative error histogram
    axes[1].hist(rel_error, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Relative Error', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Relative Error Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axvline(np.mean(rel_error), color='b', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(rel_error):.2e}')
    axes[1].legend(fontsize=10)

    if equation_name:
        fig.suptitle(f'{equation_name} - Error Distribution', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error distribution plot saved to: {save_path}")


def create_output_dirs(base_path):
    """
    Create output directories for plots

    Args:
        base_path: base directory path

    Returns:
        dict with paths to subdirectories
    """
    dirs = {
        'plots': os.path.join(base_path, 'plots'),
        'loss': os.path.join(base_path, 'plots', 'loss'),
        'solution': os.path.join(base_path, 'plots', 'solution'),
        'error': os.path.join(base_path, 'plots', 'error')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs