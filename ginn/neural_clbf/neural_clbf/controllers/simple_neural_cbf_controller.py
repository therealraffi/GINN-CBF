import itertools
from typing import Tuple, List, Optional
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
# import gurobipy as gp
# from gurobipy import GRB
import numpy as np

from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.utils import ScenarioList
from neural_clbf.controllers.cbf_controller import CBFController
from neural_clbf.controllers.controller_utils import normalize_with_angles
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite

class SimpleNeuralCBFController(CBFController):
    """
    A neural CBF controller. Differs from the CBFController in that it uses a
    neural network to learn the CBF.

    More specifically, the CBF controller looks for a V such that

    V(safe) < 0
    V(unsafe) > 0
    dV/dt <= -lambda V

    This proves forward invariance of the 0-sublevel set of V, and since the safe set is
    a subset of this sublevel set, we prove that the unsafe region is not reachable from
    the safe region.
    """

    def __init__(
        self,
        dynamics_model: ControlAffineSystem,
        scenarios,
        V_nn: nn.Module,  # Pass the neural network as an argument
        cbf_lambda: float = 1.0,
        cbf_relaxation_penalty: float = 50.0,
        small_control_penalty: float = 10.0,
        small_control_thresh: float = 0.5,
        scale_parameter: float = 10.0,
        controller_period: float = 0.01,
        disable_gurobi: bool = True,
        z = torch.tensor([1]),
        agent_rad = 0,
        device = 'cpu',
        xy_only=True
    ):
        """Initialize the controller.

        args:
            dynamics_model: the control-affine dynamics of the underlying system
            scenarios: a list of parameter scenarios to train on
            experiment_suite: defines the experiments to run during training
            cbf_hidden_layers: number of hidden layers to use for the CLBF network
            cbf_hidden_size: number of neurons per hidden layer in the CLBF network
            cbf_lambda: convergence rate for the CLBF
            cbf_relaxation_penalty: the penalty for relaxing CLBF conditions.
            controller_period: the timestep to use in simulating forward Vdot
            primal_learning_rate: the learning rate for SGD for the network weights,
                                  applied to the CLBF decrease loss
            scale_parameter: normalize non-angle data points to between +/- this value.
            learn_shape_epochs: number of epochs to spend just learning the shape
            use_relu: if True, use a ReLU network instead of Tanh
        """
        super(SimpleNeuralCBFController, self).__init__(
            dynamics_model=dynamics_model,
            scenarios=scenarios,
            experiment_suite=None,
            cbf_lambda=cbf_lambda,
            cbf_relaxation_penalty=cbf_relaxation_penalty,
            controller_period=controller_period
        )
        # self.save_hyperparameters()

        # TODO: Resolve 'clf_relaxation_penalty_weight' vs 'cbf_relaxation_penalty'
        self.clf_relaxation_penalty_weight = cbf_relaxation_penalty  # Default weight for CLF relaxation penalty
        self.small_control_penalty_weight = small_control_penalty  # Default weight for small control penalty
        self.small_control_threshold = small_control_thresh  # Threshold for small control magnitude

        self.disable_gurobi = disable_gurobi
        self.V_nn = V_nn
        self.z = z
        # self.qp_relaxation_penalty = None
        self.device = device
        self.small_control_penalty = small_control_penalty
        self.small_control_thresh = small_control_thresh
        self.lip = 0
        self.agent_rad = agent_rad

        # Save the provided model
        # self.dynamics_model = dynamics_model
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # Some of the dimensions might represent angles. We want to replace these
        # dimensions with two dimensions: sin and cos of the angle. To do this, we need
        # to figure out how many numbers are in the expanded state
        n_angles = len(self.dynamics_model.angle_dims)
        self.n_dims_extended = self.dynamics_model.n_dims + n_angles

        # Compute and save the center and range of the state variables
        x_max, x_min = dynamics_model.state_limits
        self.x_center = (x_max + x_min) / 2.0
        self.x_range = (x_max - x_min) / 2.0
        # Scale to get the input between (-k, k), centered at 0
        self.k = scale_parameter
        self.x_range = self.x_range / self.k
        # We shouldn't scale or offset any angle dimensions
        self.x_center[self.dynamics_model.angle_dims] = 0.0
        self.x_range[self.dynamics_model.angle_dims] = 1.0

        ########## optimization

        # Save the other parameters
        clf_lambda = cbf_lambda
        clf_relaxation_penalty = cbf_relaxation_penalty

        self.clf_lambda = clf_lambda
        self.safe_level: Union[torch.Tensor, float]
        self.unsafe_level: Union[torch.Tensor, float]
        self.clf_relaxation_penalty = clf_relaxation_penalty

        # Since we want to be able to solve the CLF-QP differentiably, we need to set
        # up the CVXPyLayers optimization. First, we define variables for each control
        # input and the relaxation in each scenario
        u = cp.Variable(self.dynamics_model.n_controls)
        clf_relaxations = []
        for scenario in self.scenarios:
            clf_relaxations.append(cp.Variable(1, nonneg=True))

        V_param = cp.Parameter(1, nonneg=True)
        Lf_V_params = []
        Lg_V_params = []
        for scenario in self.scenarios:
            Lf_V_params.append(cp.Parameter(1))
            Lg_V_params.append(cp.Parameter(self.dynamics_model.n_controls))

        clf_relaxation_penalty_param = cp.Parameter(1, nonneg=True)
        u_ref_param = cp.Parameter(self.dynamics_model.n_controls)
        agent_rad_param = cp.Parameter(1)
        lip_param = cp.Parameter(1)

        # These allow us to define the constraints
        constraints = []
        for i in range(len(self.scenarios)):
            constraints.append(
                Lf_V_params[i]
                + Lg_V_params[i] @ u
                >= clf_lambda * (V_param - agent_rad_param) - clf_relaxations[i] + lip_param
            )

        upper_lim, lower_lim = self.dynamics_model.control_limits
        for control_idx in range(self.dynamics_model.n_controls):
            constraints.append(u[control_idx] >= lower_lim[control_idx])
            constraints.append(u[control_idx] <= upper_lim[control_idx])
        if xy_only:
            constraints.append(u[2] == 0)

        objective_expression = cp.norm(u - u_ref_param, p=2)
        for r in clf_relaxations:
            objective_expression += cp.multiply(clf_relaxation_penalty_param, r)
        objective = cp.Minimize(objective_expression)

        for r in clf_relaxations:
            objective_expression += cp.multiply(clf_relaxation_penalty_param, r)

        problem = cp.Problem(objective, constraints)
        assert problem.is_dpp()
        variables = [u] + clf_relaxations
        parameters = Lf_V_params + Lg_V_params
        parameters += [V_param, u_ref_param, clf_relaxation_penalty_param, agent_rad_param, lip_param]
        self.differentiable_qp_solver = CvxpyLayer(
            problem, variables=variables, parameters=parameters
        )

    @property
    def cbf_lambda(self):
        """Rename clf lambda to cbf"""
        return self.clf_lambda

    def set_V_nn(self, V_nn: nn.Module):
        self.V_nn = V_nn

    def set_lipshitz(self, val):
        self.lip = val

    def _solve_CLF_QP_cvxpylayers(
            self,
            x: torch.Tensor,
            u_ref: torch.Tensor,
            V: torch.Tensor,
            Lf_V: torch.Tensor,
            Lg_V: torch.Tensor,
            relaxation_penalty: float,
            lip = 0,
            agent_rad = 0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Determine the control input for a given state using a QP. Solves the QP using
            CVXPyLayers, which does allow for backpropagation, but is slower and less
            accurate than Gurobi.

            args:
                x: bs x self.dynamics_model.n_dims tensor of state
                u_ref: bs x self.dynamics_model.n_controls tensor of reference controls
                V: bs x 1 tensor of CLF values,
                Lf_V: bs x 1 tensor of CLF Lie derivatives,
                Lg_V: bs x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
                relaxation_penalty: the penalty to use for CLF relaxation.
            returns:
                u: bs x self.dynamics_model.n_controls tensor of control inputs
                relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                            case
            """

            # print(x.shape)
            # print(u_ref.shape)
            # print(V.shape)
            # print(Lf_V.shape)
            # print(Lg_V.shape)
            # print(relaxation_penalty)

            relaxation_penalty = min(relaxation_penalty, 1e6)

            # print("l", lip.reshape(-1, 1).shape)
            # print("v", V.reshape(-1, 1).shape)

            params = []
            for i in range(self.n_scenarios):
                params.append(Lf_V[:, i, :])
            for i in range(self.n_scenarios):
                params.append(Lg_V[:, i, :])
            params.append(V.reshape(-1, 1))
            params.append(u_ref)
            params.append(torch.tensor([relaxation_penalty]).type_as(x))
            params.append(lip.reshape(-1, 1).type_as(x))
            params.append(torch.tensor([agent_rad]).type_as(x))

            # print()
            # print(x)
            # print(u_ref)
            # print(V)
            # print(Lf_V)
            # print(Lg_V)

            result = self.differentiable_qp_solver(
                *params,
                solver_args={"max_iters": 1500},
            )

            u_result = result[0]
            r_result = torch.hstack(result[1:])

            # print("res", u_result.type_as(x))

            return u_result.type_as(x), r_result.type_as(x)

    def solve_CLF_QP(
        self,
        x,
        relaxation_penalty: Optional[float] = None,
        u_ref: Optional[torch.Tensor] = None,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            relaxation_penalty: the penalty to use for CLF relaxation, defaults to
                                self.clf_relaxation_penalty
            u_ref: allows the user to supply a custom reference input, which will
                   bypass the self.u_reference function. If provided, must have
                   dimensions bs x self.dynamics_model.n_controls. If not provided,
                   default to calling self.u_reference.
            requires_grad: if True, use a differentiable layer
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """

        V = self.V(x).to(self.device)
        Lf_V, Lg_V = self.V_lie_derivatives(x)

        if u_ref is not None:
            err_message = f"u_ref must have {x.shape[0]} rows, but got {u_ref.shape[0]}"
            assert u_ref.shape[0] == x.shape[0], err_message
            err_message = f"u_ref must have {self.dynamics_model.n_controls} cols,"
            err_message += f" but got {u_ref.shape[1]}"
            assert u_ref.shape[1] == self.dynamics_model.n_controls, err_message
        else:
            u_ref = self.u_reference(x)

        if relaxation_penalty is None:
            relaxation_penalty = self.clf_relaxation_penalty
        
        lip = self.lip
        agent_rad = self.agent_rad

        # print(V.device)
        return self._solve_CLF_QP_cvxpylayers(
            x, u_ref, V, Lf_V, Lg_V, relaxation_penalty, lip, agent_rad
        )

    def V_with_jacobian(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        new_shape = x.shape[:-1] + (1,)
        z = torch.full(new_shape, self.z.item()).to(self.device)

        out = self.V_nn(x, z).to(self.device)
        jac = self.V_nn.get_jacobian(x, z).to(self.device)
        return out, jac

    def descent_loss(
        self,
        x: torch.Tensor,
        requires_grad: bool = False,
        u_ref = None
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CBF due to the descent condition.

        args:
            x: the points at which to evaluate the loss.
            goal_mask: the points in x marked as part of the goal.
            safe_mask: the points in x marked safe.
            unsafe_mask: the points in x marked unsafe.
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses.
            requires_grad: if True, use a differentiable QP solver.
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss to encourage satisfaction of the descent condition.
        loss = []

        # Get the control input and relaxation from solving the QP, and aggregate
        # the relaxation across scenarios.
        u_qp, qp_relaxation = self.solve_CLF_QP(x, requires_grad=requires_grad, u_ref = u_ref)
        qp_relaxation = torch.mean(qp_relaxation, dim=-1)

        # Minimize the QP relaxation to encourage satisfying the decrease condition.
        qp_relaxation_loss = qp_relaxation.mean()
        loss.append(("QP relaxation", qp_relaxation_loss))

        return loss
    
    def get_optimal_control(
        self,
        x: torch.Tensor,
        requires_grad: bool = False,
        u_ref = None
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CBF due to the descent condition.

        args:
            x: the points at which to evaluate the loss.
            goal_mask: the points in x marked as part of the goal.
            safe_mask: the points in x marked safe.
            unsafe_mask: the points in x marked unsafe.
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses.
            requires_grad: if True, use a differentiable QP solver.
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss to encourage satisfaction of the descent condition.
        loss = []

        # Get the control input and relaxation from solving the QP, and aggregate
        # the relaxation across scenarios.
        # dev = self.device
        # self.device = 'cpu'
        u_qp, qp_relaxation = self.solve_CLF_QP(x, requires_grad=requires_grad, u_ref=u_ref)
        # self.device = dev
        qp_relaxation = torch.mean(qp_relaxation, dim=-1)
        self.qp_relaxation_penalty = qp_relaxation

        return u_qp
  
    def V_lie_derivatives(
        self, x: torch.Tensor, scenarios = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the Lie derivatives of the CLF V along the control-affine dynamics

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            scenarios: optional list of scenarios. Defaults to self.scenarios
        returns:
            Lf_V: bs x len(scenarios) x 1 tensor of Lie derivatives of V
                  along f
            Lg_V: bs x len(scenarios) x self.dynamics_model.n_controls tensor
                  of Lie derivatives of V along g
        """
        if scenarios is None:
            scenarios = self.scenarios
        n_scenarios = len(scenarios)

        # Get the Jacobian of V for each entry in the batch
        _, gradV = self.V_with_jacobian(x)
        gradV = gradV.to(self.device) # N x 3
        gradV = gradV.unsqueeze(1) # N x 3 x 1

        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, n_scenarios, 1)
        Lg_V = torch.zeros(batch_size, n_scenarios, self.dynamics_model.n_controls)
        Lf_V = Lf_V.type_as(x)
        Lg_V = Lg_V.type_as(x)

        for i in range(n_scenarios):
            # Get the dynamics f and g for this scenario
            s = scenarios[i]
            f, g = self.dynamics_model.control_affine_dynamics(x, params=s)
            f = f.to(self.device) # N x 3 x 1
            g = g.to(self.device)
            
            Lf_V[:, i, :] = torch.bmm(gradV, f).squeeze(1)
            Lg_V[:, i, :] = torch.bmm(gradV, g).squeeze(1)

        # return the Lie derivatives
        return Lf_V, Lg_V
