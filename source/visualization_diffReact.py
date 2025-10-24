"""
comparative_viz.py: Comparative Visualization utilities for PINN vs ST-PINN

This library provides functions to compare the training metrics and final solution
quality of a standard Physics-Informed Neural Network (PINN) and a Self-Training
Physics-Informed Neural Network (ST-PINN).

ASSUMPTION: The training scripts (e.g., DiffReact1D_PINN.py and DiffReact1D_STPINN.py)
have been modified to save structured history and prediction data into .npy files.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_comparative_loss(loss_history_pinn, loss_history_stpinn, save_path):
    """
    Compares the total loss and equation residual loss for PINN and ST-PINN.

    Args:
        loss_history_pinn (dict): dict of loss history for PINN
        loss_history_stpinn (dict): dict of loss history for ST-PINN
        save_path (str): path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    iterations_pinn = loss_history_pinn['iteration']
    iterations_stpinn = loss_history_stpinn['iteration']

    # --- Total Loss Comparison (Left) ---
    ax0 = axes[0]
    ax0.semilogy(iterations_pinn, loss_history_pinn['total_loss'], 'b-', linewidth=2, label='PINN Total Loss')
    ax0.semilogy(iterations_stpinn, loss_history_stpinn['total_loss'], 'r--', linewidth=2, label='ST-PINN Total Loss')
    ax0.set_xlabel('Iteration', fontsize=12)
    ax0.set_ylabel('Loss', fontsize=12)
    ax0.set_title('Total Training Loss Comparison', fontsize=14, fontweight='bold')
    ax0.legend(fontsize=10)
    ax0.grid(True, alpha=0.3)

    # --- Equation Residual Loss Comparison (Right) ---
    ax1 = axes[1]
    ax1.semilogy(iterations_pinn, loss_history_pinn['eqns_loss'], 'b-', linewidth=2, label='PINN $\mathcal{L}_{r}$ Loss')
    ax1.semilogy(iterations_stpinn, loss_history_stpinn['eqns_loss'], 'r--', linewidth=2, label='ST-PINN $\mathcal{L}_{r}$ Loss')

    # Optionally plot Pseudo-Label Loss for context
    if 'pseudo_loss' in loss_history_stpinn and len(loss_history_stpinn['pseudo_loss']) > 0:
        ax1.semilogy(iterations_stpinn, loss_history_stpinn['pseudo_loss'], 'g:', linewidth=1.5, label='ST-PINN Pseudo-Label $\mathcal{L}_{p}$')

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('PDE Residual Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparative loss plot saved to: {save_path}")


def plot_comparative_error_history(error_history_pinn, error_history_stpinn, save_path):
    """
    Compares the relative L2 test error history for PINN and ST-PINN.

    Args:
        error_history_pinn (dict): dict of error history for PINN
        error_history_stpinn (dict): dict of error history for ST-PINN
        save_path (str): path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(error_history_pinn['iteration'], error_history_pinn['error'], 'b-',
                linewidth=2, marker='o', markersize=4, label='PINN Relative L2 Error')
    ax.semilogy(error_history_stpinn['iteration'], error_history_stpinn['error'], 'r--',
                linewidth=2, marker='s', markersize=4, label='ST-PINN Relative L2 Error')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Relative $\mathcal{L}_2$ Error', fontsize=12)
    ax.set_title('Test Error History Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparative error plot saved to: {save_path}")


def plot_comparative_error_maps_1d(x, t, u_true, u_pred_pinn, u_pred_stpinn, save_path, equation_name=""):
    """
    Compares the 2D spatio-temporal absolute error maps for PINN and ST-PINN.

    Args:
        x, t (numpy.ndarray): coordinates
        u_true (numpy.ndarray): ground truth solution
        u_pred_pinn (numpy.ndarray): PINN predicted solution
        u_pred_stpinn (numpy.ndarray): ST-PINN predicted solution
        save_path (str): path to save the figure
        equation_name (str): name of the equation for the title
    """
    # Flatten arrays
    x = x.flatten()
    t = t.flatten()
    u_true = u_true.flatten()
    u_pred_pinn = u_pred_pinn.flatten()
    u_pred_stpinn = u_pred_stpinn.flatten()

    # Create grid for plotting
    x_unique = np.unique(x)
    t_unique = np.unique(t)

    # Interpolate solutions onto grid
    X, T = np.meshgrid(x_unique, t_unique)
    U_true = griddata((x, t), u_true, (X, T), method='linear')
    U_pred_pinn = griddata((x, t), u_pred_pinn, (X, T), method='linear')
    U_pred_stpinn = griddata((x, t), u_pred_stpinn, (X, T), method='linear')

    # Compute errors
    error_pinn = np.abs(U_true - U_pred_pinn)
    error_stpinn = np.abs(U_true - U_pred_stpinn)

    # --- Create Figure: PINN Error, ST-PINN Error, and Difference ---
    fig = plt.figure(figsize=(18, 5))
    gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)

    # Set common max error scale for error maps
    vmax_common = max(np.nanmax(error_pinn), np.nanmax(error_stpinn))

    # --- 1. PINN Absolute Error ---
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(error_pinn, aspect='auto', cmap='hot',
                     extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
                     origin='lower', interpolation='bilinear', vmax=vmax_common)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('t', fontsize=12)
    ax1.set_title('PINN Absolute Error', fontsize=13, fontweight='bold')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im1, cax=cax1)

    # --- 2. ST-PINN Absolute Error ---
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(error_stpinn, aspect='auto', cmap='hot',
                     extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
                     origin='lower', interpolation='bilinear', vmax=vmax_common)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('t', fontsize=12)
    ax2.set_title('ST-PINN Absolute Error', fontsize=13, fontweight='bold')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im2, cax=cax2)

    # --- 3. Error Difference (PINN - ST-PINN) ---
    ax3 = fig.add_subplot(gs[2])
    # Positive values: PINN error is larger (ST-PINN is better)
    # Negative values: ST-PINN error is larger (PINN is better)
    error_diff = error_pinn - error_stpinn
    vmax_abs = np.nanmax(np.abs(error_diff))

    im3 = ax3.imshow(error_diff, aspect='auto', cmap='seismic',
                     extent=[x_unique.min(), x_unique.max(), t_unique.min(), t_unique.max()],
                     origin='lower', interpolation='bilinear', vmin=-vmax_abs, vmax=vmax_abs)
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('t', fontsize=12)
    ax3.set_title('Error Difference (PINN - ST-PINN)', fontsize=13, fontweight='bold')
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('Positive = ST-PINN better')

    if equation_name:
        fig.suptitle(f'{equation_name} - Absolute Error Comparison', fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparative error map plot saved to: {save_path}")