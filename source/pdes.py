import numpy as np
import torch


def gradient(y, x):
    """
    Compute gradient of y with respect to x using PyTorch autograd.

    Args:
        y: Output tensor
        x: Input tensor

    Returns:
        Gradient dy/dx
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
    return grad


def Burgers1D(x, t, u, nu):
    """
    1D Burgers equation: u_t + u*u_x - (nu/pi)*u_xx = 0

    Args:
        x: Spatial coordinate tensor (requires_grad=True)
        t: Time coordinate tensor (requires_grad=True)
        u: Solution tensor from neural network
        nu: Viscosity parameter

    Returns:
        PDE residual
    """
    u_t = gradient(u, t)
    u_x = gradient(u, x)
    u_xx = gradient(u_x, x)

    e = u_t + u * u_x - nu / np.pi * u_xx
    return e


def diffusion_reaction_1d(x, t, u, nu, rho):
    """
    1D Diffusion-Reaction equation: u_t - nu*u_xx - rho*u*(1-u) = 0

    Args:
        x: Spatial coordinate tensor (requires_grad=True)
        t: Time coordinate tensor (requires_grad=True)
        u: Solution tensor from neural network
        nu: Diffusion coefficient
        rho: Reaction rate

    Returns:
        PDE residual
    """
    u_t = gradient(u, t)
    u_x = gradient(u, x)
    u_xx = gradient(u_x, x)

    e = u_t - nu * u_xx - rho * (u - u * u)
    return e


def Diffusion_sorption(x, t, u):
    """
    1D Diffusion-Sorption equation with retardation factor.

    Args:
        x: Spatial coordinate tensor (requires_grad=True)
        t: Time coordinate tensor (requires_grad=True)
        u: Concentration tensor from neural network

    Returns:
        PDE residual
    """
    # Physical parameters
    D = 5e-4  # Diffusion coefficient
    por = 0.29  # Porosity
    rho_s = 2880  # Solid density
    k_f = 3.5e-4  # Freundlich constant
    n_f = 0.874  # Freundlich exponent

    # Retardation factor
    retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * (u + 1e-6) ** (n_f - 1)

    u_t = gradient(u, t)
    u_x = gradient(u, x)
    u_xx = gradient(u_x, x)

    e = u_t - D / retardation_factor * u_xx
    return e


def Boundary(x, u):
    """
    Boundary condition for diffusion-sorption equation.

    Args:
        x: Spatial coordinate tensor (requires_grad=True)
        u: Concentration tensor from neural network

    Returns:
        Boundary flux: D * u_x
    """
    D = 5e-4  # Diffusion coefficient
    u_x = gradient(u, x)

    return D * u_x


def SWE_2D(u, v, h, x, y, t, g):
    """
    2D Shallow Water Equations.

    Continuity: h_t + (hu)_x + (hv)_y = 0
    Momentum X: u_t + u*u_x + v*u_y + g*h_x = 0
    Momentum Y: v_t + u*v_x + v*v_y + g*h_y = 0

    Args:
        u: X-velocity tensor
        v: Y-velocity tensor
        h: Water height tensor
        x: X-coordinate tensor (requires_grad=True)
        y: Y-coordinate tensor (requires_grad=True)
        t: Time coordinate tensor (requires_grad=True)
        g: Gravitational acceleration

    Returns:
        Tuple of three residuals (e1, e2, e3)
    """
    u_t = gradient(u, t)
    v_t = gradient(v, t)
    h_t = gradient(h, t)

    u_x = gradient(u, x)
    v_x = gradient(v, x)
    h_x = gradient(h, x)

    u_y = gradient(u, y)
    v_y = gradient(v, y)
    h_y = gradient(h, y)

    e1 = h_t + h_x * u + h * u_x + h_y * v + h * v_y
    e2 = u_t + u * u_x + v * u_y + g * h_x
    e3 = v_t + u * v_x + v * v_y + g * h_y

    return e1, e2, e3


def Boundary_condition(x, y, u, v):
    """
    Boundary conditions for 2D flow (zero normal derivatives).

    Args:
        x: X-coordinate tensor (requires_grad=True)
        y: Y-coordinate tensor (requires_grad=True)
        u: X-velocity tensor
        v: Y-velocity tensor

    Returns:
        Tuple of gradients (u_x, v_x, u_y, v_y)
    """
    u_x = gradient(u, x)
    v_x = gradient(v, x)
    u_y = gradient(u, y)
    v_y = gradient(v, y)

    return u_x, v_x, u_y, v_y


def CFD_2D(x, y, t, d, u, v, p, gamma, keci, yifu):
    """
    2D Compressible Fluid Dynamics equations (Euler/Navier-Stokes).

    Args:
        x: X-coordinate tensor (requires_grad=True)
        y: Y-coordinate tensor (requires_grad=True)
        t: Time coordinate tensor (requires_grad=True)
        d: Density tensor
        u: X-velocity tensor
        v: Y-velocity tensor
        p: Pressure tensor
        gamma: Specific heat ratio
        keci: Dynamic viscosity
        yifu: Bulk viscosity

    Returns:
        Tuple of four residuals (e1, e2, e3, e4) for conservation of mass, x-momentum, y-momentum, and energy
    """
    # Energy and flux terms
    E = p / (gamma - 1.0) + 0.5 * d * (u ** 2 + v ** 2)
    Fu = u * (E + p)
    Fv = v * (E + p)
    du = d * u
    dv = d * v

    # First-order derivatives
    d_t = gradient(d, t)
    du_x = gradient(du, x)
    dv_y = gradient(dv, y)
    u_t = gradient(u, t)
    u_x = gradient(u, x)
    u_y = gradient(u, y)
    v_t = gradient(v, t)
    v_x = gradient(v, x)
    v_y = gradient(v, y)
    p_x = gradient(p, x)
    p_y = gradient(p, y)
    E_t = gradient(E, t)
    Fu_x = gradient(Fu, x)
    Fv_y = gradient(Fv, y)

    # Second-order derivatives for viscous terms
    u_xx = gradient(u_x, x)
    v_xx = gradient(v_x, x)
    u_yy = gradient(u_y, y)
    v_yy = gradient(v_y, y)
    v_yx = gradient(v_y, x)
    u_xy = gradient(u_x, y)

    # Conservation equations
    e1 = d_t + du_x + dv_y  # Mass conservation
    e2 = d * (u_t + u * u_x + v * u_y) + p_x - keci * (u_xx + u_yy) - (keci + yifu / 3.0) * (u_xx + v_yx)  # X-momentum
    e3 = d * (v_t + u * v_x + v * v_y) + p_y - keci * (v_xx + v_yy) - (keci + yifu / 3.0) * (u_xy + v_yy)  # Y-momentum
    e4 = E_t + Fu_x + Fv_y  # Energy conservation

    return e1, e2, e3, e4