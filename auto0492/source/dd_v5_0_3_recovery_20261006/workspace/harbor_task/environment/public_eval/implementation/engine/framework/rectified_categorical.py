"""Offline categorical rectified flow; DD training reimplementation support.

This is not released DD training code. No model or data is accessed here.
The source-to-data time is tau; DPM-Solver uses s=1-tau instead.
"""
import torch
from vendor.dpm_solver_pytorch import DPM_Solver


class RectifiedFlowSchedule:
    T = 1.0
    total_N = 1000

    def marginal_log_mean_coeff(self, s):
        return torch.log1p(-s)

    def marginal_alpha(self, s):
        return 1 - s

    def marginal_std(self, s):
        return s

    def marginal_lambda(self, s):
        return torch.log1p(-s) - torch.log(s)

    def inverse_lambda(self, value):
        return torch.sigmoid(-value)


class CategoricalFlow:
    """Analytic Gaussian-mixture posterior for a fixed categorical condition.

    x_tau = (1-tau)*epsilon + tau*c_j, c_j drawn with the teacher probabilities.
    The posterior mean is sum_j softmax(logp_j - ||x-tau*c_j||^2 /
    (2*(1-tau)^2))*c_j. A matrix product avoids a rows*vocab*channels tensor.
    """

    def __init__(self, codebook, log_probabilities, chunk_rows=2048):
        if codebook.ndim != 2 or log_probabilities.shape[-1] != len(codebook):
            raise ValueError('Incompatible codebook and categorical probabilities')
        if chunk_rows < 1:
            raise ValueError('chunk_rows must be positive')
        self.codebook = codebook.detach()
        self.logp = torch.log_softmax(log_probabilities.detach(), dim=-1)
        self.codebook_norm = self.codebook.square().sum(dim=-1)
        self.chunk_rows = chunk_rows
        self.evaluations = 0

    def posterior_mean(self, x, tau):
        if x.shape[:-1] != self.logp.shape[:-1] or x.shape[-1] != self.codebook.shape[-1]:
            raise ValueError('x and conditional probabilities must have matching rows')
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        tau = torch.as_tensor(tau, dtype=dtype, device=x.device).reshape(())
        if not 0 <= float(tau) < 1:
            raise ValueError('posterior requires 0 <= tau < 1')
        book = self.codebook.to(dtype=dtype, device=x.device)
        norm = self.codebook_norm.to(dtype=dtype, device=x.device)
        flat_x = x.to(dtype).reshape(-1, x.shape[-1])
        flat_logp = self.logp.to(dtype=dtype, device=x.device).reshape(-1, len(book))
        parts = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for begin in range(0, len(flat_x), self.chunk_rows):
                chunk = flat_x[begin:begin + self.chunk_rows]
                # The term ||x||^2 is independent of j and cancels in softmax.
                distance = tau.square() * norm - 2 * tau * (chunk @ book.T)
                logits = flat_logp[begin:begin + len(chunk)] - distance / (2 * (1 - tau).square())
                parts.append(logits.softmax(dim=-1) @ book)
        self.evaluations += 1
        return torch.cat(parts).reshape_as(x).to(x.dtype)

    def velocity(self, x, tau):
        return (self.posterior_mean(x, tau) - x) / (1 - tau)

    def nearest_indices(self, x):
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        book = self.codebook.to(dtype=dtype, device=x.device)
        norm = self.codebook_norm.to(dtype=dtype, device=x.device)
        flat = x.to(dtype).reshape(-1, x.shape[-1])
        parts = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for begin in range(0, len(flat), self.chunk_rows):
                # ||x||^2 is independent of the candidate codebook index.
                distance = norm - 2 * (flat[begin:begin + self.chunk_rows] @ book.T)
                parts.append(distance.argmin(dim=-1))
        return torch.cat(parts).reshape(x.shape[:-1])

    @torch.no_grad()
    def sample_endpoint(self, noise, nfe=10, boundary_epsilon=1e-3, skip_type='time_uniform'):
        if nfe < 3 or not 0 < boundary_epsilon < .05:
            raise ValueError('Invalid solver budget or boundary epsilon')
        if skip_type not in ('time_uniform', 'logSNR'):
            raise ValueError('Unsupported solver time grid')
        schedule = RectifiedFlowSchedule()
        flow = self

        class PosteriorSolver(DPM_Solver):
            # Use the analytic data predictor directly to avoid subtraction
            # cancellation from converting it to epsilon and back near tau=0.
            def data_prediction_fn(self, x, s):
                return flow.posterior_mean(x, 1 - s)

        solver = PosteriorSolver(lambda x, s: x, schedule, algorithm_type='dpmsolver++')
        prior_mean = self.logp.exp() @ self.codebook
        # The first-order startup is explicit: tau=0 is a singular log-SNR.
        x = (1 - boundary_epsilon) * noise + boundary_epsilon * prior_mean
        self.evaluations = 0
        x = solver.sample(x, steps=nfe, t_start=1 - boundary_epsilon,
                          t_end=boundary_epsilon, order=3, skip_type=skip_type,
                          method='multistep', lower_order_final=True,
                          denoise_to_zero=False)
        if self.evaluations != nfe:
            raise RuntimeError('Actual posterior NFE differs from the requested budget')
        if not torch.isfinite(x).all():
            raise FloatingPointError('Nonfinite categorical FM endpoint')
        return x

    @torch.no_grad()
    def sample_indices(self, noise, nfe=10, boundary_epsilon=1e-3, skip_type='time_uniform'):
        return self.nearest_indices(self.sample_endpoint(noise, nfe, boundary_epsilon, skip_type))


def uniform_to_scale(t, num_scales=10):
    """Freeze the explicit [0,1] -> 0..9 prefix mapping for DD training.

    The endpoint 1 maps to the last valid scale. No candidate distribution
    is defined here; this adapter consumes method.py's sampler output.
    """
    if not torch.isfinite(t).all() or not ((t >= 0) & (t <= 1)).all():
        raise ValueError('Method time sampler must return finite values in [0,1]')
    return torch.floor(t * num_scales).long().clamp_max(num_scales - 1)
