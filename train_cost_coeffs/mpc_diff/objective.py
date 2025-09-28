import torch


class QuadraticCost(torch.nn.Module):
    """
    Quadratic Objective for diff-mpc.\n
    iLQR can have non-linear cost as well.\n
    Final cost is defined implicitly using vectors q, p.\n
    Where q is diagonal of C_t and p is c_t of cost.\n
    - q: (n_state + n_ctrl,) diag entries\n
    - p: (n_state + n_ctrl,) linear term
    -----------
    - Cost co-efficients are learnable (state co-effs, control co-effs are fixed)
    - Co-efficients are represented as `co-effs = softplus(raw_coeff)`
    """

    target_state: torch.Tensor

    def __init__(
        self,
        n_state: int,
        n_ctrl: int,
        target_state: torch.Tensor,
        state_coeff_init: float,  # =0.1,
        ctrl_coeff: float,  # =0.001,
        device="cpu",
    ):
        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.ctrl_coeffs = QuadraticCost.get_ctrl_coeffs(n_ctrl, ctrl_coeff, device)
        self.target_ctrl_with_coeffs = torch.zeros(
            self.n_ctrl, dtype=torch.float32, device=device
        )

        state_coeffs = torch.full(
            (n_state,), state_coeff_init, dtype=torch.float32, device=device
        )
        state_coeffs_raw = torch.log(torch.expm1(state_coeffs.clamp(min=1e-6)))
        self.state_coeffs_raw = torch.nn.Parameter(state_coeffs_raw)

        self.register_buffer("target_state", target_state.clone().detach().to(device))

    def state_coeffs(self):
        # softplus ensures positive weights
        return torch.nn.functional.softplus(self.state_coeffs_raw) + 1e-8

    def forward(self):
        """
        Returns q, p vectors
        """
        return QuadraticCost.get_quadratic_objective(
            self.state_coeffs(), self.ctrl_coeffs, self.target_state
        )

    @staticmethod
    def get_ctrl_coeffs(n_ctrl, coeff_val, device):
        return coeff_val * torch.ones(n_ctrl, dtype=torch.float32, device=device)

    @staticmethod
    def get_quadratic_objective(state_coeffs, ctrl_coeffs, target_state):
        coeffs = torch.cat((state_coeffs, ctrl_coeffs), dim=0)

        target_ctrl_with_coeffs = torch.zeros(
            len(ctrl_coeffs), dtype=torch.float32, device=coeffs.device
        )

        target_state_with_coeffs = -torch.clamp(state_coeffs, min=1e-12) * target_state
        # target_state_with_coeffs = (
        #   -torch.sqrt(torch.clamp(state_coeffs, min=1e-12)) * target_state
        # )

        target_with_coeffs = torch.cat(
            (target_state_with_coeffs, target_ctrl_with_coeffs), dim=0
        )

        return coeffs, target_with_coeffs  # (q, p) or (C, c)


class ExpertQuadraticCost(torch.nn.Module):

    target_state: torch.Tensor
    target_state_coeffs: torch.Tensor

    def __init__(
        self,
        n_state: int,
        n_ctrl: int,
        target_state: torch.Tensor,
        target_state_coeffs: torch.Tensor,
        ctrl_coeff: int,
        device="cpu",
    ):
        """
        Expert quadratic cost, this does not have any learnable coeffs, rather its used
        to create expert data for training
        ----------
        - `get_true_objective` returns the q, p vecs based on goal state
        and weights defined in class initialization
        """

        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.ctrl_penalty = ctrl_coeff

        self.register_buffer("target_state", target_state)
        self.register_buffer("target_state_coeffs", target_state_coeffs)
        self.ctrl_coeffs = QuadraticCost.get_ctrl_coeffs(n_ctrl, ctrl_coeff, device)

    def get_true_objective(self):
        """
        Gives linear quadratic cost based on the defined goal states and their weights

        Returns (q, p) vectors where q is diagonal of C_t and p is c_t of cost:
          q: (n_state + n_ctrl,) diag entries
          p: (n_state + n_ctrl,) linear term
        """
        return QuadraticCost.get_quadratic_objective(
            self.target_state_coeffs, self.ctrl_coeffs, self.target_state
        )
