from typing import Tuple, List, Optional

import torch
import numpy as np
from .control_affine_system import ControlAffineSystem

class Simple3DRobot(ControlAffineSystem):
    """
    Represents a minimal 3D robot system.

    The system has state:
        x = [px, py, pz]

    representing the position of the robot in 3D space, and control inputs:
        u = [vx, vy, vz]

    representing the velocities in each dimension.
    """

    # Number of states and controls
    N_DIMS = 3
    N_CONTROLS = 3

    # State indices
    PX = 0
    PY = 1
    PZ = 2
    # PZZ = 3

    # Control indices
    VX = 0
    VY = 1
    VZ = 2
    # VZZ = 3

    def __init__(
        self,
        nominal_params: dict = None,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios = None
    ):
        """
        Initialize the robot system.

        args:
            nominal_params: optional parameters for the system (not used here).
            dt: the timestep to use for the simulation.
            controller_dt: the timestep for the LQR discretization. Defaults to dt.
        """
        super().__init__(nominal_params or {}, dt, controller_dt)

    def validate_params(self, params: dict) -> bool:
        """Check if a given set of parameters is valid."""
        # No specific parameters to validate for this system.
        return True

    @property
    def n_dims(self) -> int:
        return Simple3DRobot.N_DIMS
    
    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return Simple3DRobot.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system.
        """
        upper_limit = torch.tensor([50.0, 50.0, 50.0])
        lower_limit = -upper_limit
        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system.
        """
        upper_limit = torch.tensor([15.0, 15.0, 15.0])
        lower_limit = -upper_limit
        return (upper_limit, lower_limit)

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating safe regions."""
        # Assume the entire space is safe for simplicity.
        return torch.ones_like(x[:, 0], dtype=torch.bool)

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating unsafe regions."""
        # Assume no unsafe regions for simplicity.
        return torch.zeros_like(x[:, 0], dtype=torch.bool)

    def goal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mask of x indicating points in the goal set."""
        goal_radius = 0.2
        goal_center = torch.tensor([0.0, 0.0, 0.0])
        goal_mask = (x - goal_center).norm(dim=-1) <= goal_radius
        return goal_mask

    def _f(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Return the control-independent part of the dynamics.

        args:
            x: bs x self.n_dims tensor of state.
            params: parameters for the system (not used here).
        returns:
            f: bs x self.n_dims x 1 tensor.
        """
        # No drift term for this system (stationary when no control input).
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1), device=x.device, dtype=x.dtype)
        return f

    def _g(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Return the control-dependent part of the dynamics.

        args:
            x: bs x self.n_dims tensor of state.
            params: parameters for the system (not used here).
        returns:
            g: bs x self.n_dims x self.n_controls tensor.
        """
        batch_size = x.shape[0]
        mat = torch.eye(self.n_dims)
        # mat[Simple3DRobot.PZZ, :] = 0
        g = mat.expand(batch_size, -1, -1)
        return g

    @property
    def u_eq(self) -> torch.Tensor:
        """Return the equilibrium control input (zero for this system)."""
        return torch.zeros((1, self.n_controls))

    def next_state(self, x: torch.Tensor, u: torch.Tensor, params: dict = None) -> torch.Tensor:
        # Get the control-independent and control-dependent dynamics
        f = self._f(x, params)  # bs x n_dims x 1
        g = self._g(x, params)  # bs x n_dims x n_controls
        
        dx = f.squeeze(-1) + torch.bmm(g, u.unsqueeze(-1)).squeeze(-1)  # bs x n_dims
        next_x = x + self.dt * dx  # bs x n_dims

        # print("dx", dx, 'dt', self.dt, 'u', u, 'state', next_x)
        return next_x

