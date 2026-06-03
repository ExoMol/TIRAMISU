import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)


@dataclass
class AccelerationConfig:
    """Configuration for acceleration methods."""
    # Startup behaviour.
    warmup_iterations: int = 5  # No acceleration for first N iterations.

    # Adaptive damping
    omega_start: float = 1.0
    omega_min: float = 0.5
    omega_max: float = 1.0
    omega_increase_factor: float = 1.1  # Recovering from oscillation; return preference to new step.
    omega_decrease_factor: float = 0.7  # Oscillating; prefer pops from previous iteration.

    # Ng acceleration
    ng_history_length: int = 4
    ng_enable_threshold: float = 5e-3  # 0.5% - turn on when converging smoothly.
    ng_disable_iterations: int = 5  # Disable for N iters after failure.

    ng_regularization: float = 1e-6
    ng_max_condition: float = 1e10
    ng_max_alpha: float = 3.0
    ng_step_limit: float = 2.0
    ng_worsening_factor: float = 1.2
    min_relative_pop: float = 1e-70
    ng_mix: float = 0.5
    ng_omega_cap = 0.8

    # Convergence
    convergence_threshold: float = 1e-3  # 0.1%.

    # Safety
    check_max_jump: bool = True
    max_jump_factor: float = 5.0
    min_pop_for_jump_check: float = 1e-6


class LayerAccelerator:
    """
    Accelerator for a single layer.

    Tracks history and applies acceleration/damping as needed.
    """

    __slots__ = ["layer_idx", "config", "change_history", "ng_history", "omega", "ng_disabled_until_iter",
                 "last_iteration", "residual_history"]

    def __init__(self, layer_idx: int, config: AccelerationConfig):
        self.layer_idx = layer_idx
        self.config = config

        # History storage - grows dynamically as iterations proceed
        # Each element is the max change for that iteration
        self.change_history = []  # List of floats, one per iteration

        # Ng history - stores population arrays
        self.ng_history = []  # List of arrays, one per iteration
        self.residual_history = []

        # State
        self.omega = config.omega_start
        self.ng_disabled_until_iter = -1
        self.last_iteration = 0

    def update(
            self, pop_new: npt.NDArray[np.float64], pop_old: npt.NDArray[np.float64], iteration: int
    ) -> npt.NDArray[np.float64]:
        """
        Apply Ng/DIIS acceleration or damping for this layer, dependent on several convergence criteria checks.

        Parameters
        ----------
        pop_new : ndarray
            Newly solved populations.
        pop_old : ndarray
            Previous iteration populations.
        iteration : int
            Current iteration number (0-indexed).

        Returns
        -------
            Accelerated/damped populations.
        """
        residual = pop_new - pop_old
        max_change = self._compute_change(pop_new, pop_old)

        # Store convergence history only.
        if iteration >= len(self.change_history):
            self.change_history.append(max_change)
        else:
            self.change_history[iteration] = max_change

        # Warmup phase.
        if iteration < self.config.warmup_iterations:
            accepted = pop_new.copy()

            self._store_iteration(accepted, residual)
            self.last_iteration = iteration

            return accepted

        # Always compute damped baseline first.
        pop_damped = self._apply_damping(pop_new, pop_old)
        accepted = pop_damped

        # Attempt Ng/DIIS.
        if self._should_use_ng(iteration):
            try:
                pop_ng_raw = self._apply_ng(pop_new)
                # Ng damping.
                pop_ng = (
                        self.config.ng_mix * pop_ng_raw
                        + (1.0 - self.config.ng_mix) * pop_damped
                )
                pop_ng /= pop_ng.sum()

                if self._accept_ng(
                        pop_ng=pop_ng,
                        pop_damped=pop_damped,
                        pop_old=pop_old,
                ):
                    accepted = pop_ng
                    log.info(f"[nL{self.layer_idx}] Accepted Ng acceleration.")
                else:
                    log.debug(f"[nL{self.layer_idx}] Rejected Ng acceleration.")
                    self.ng_disabled_until_iter = iteration + self.config.ng_disable_iterations

            except RuntimeError as e:
                log.warning(f"[nL{self.layer_idx}] Ng failed: {e}.")
                self.ng_disabled_until_iter = iteration + self.config.ng_disable_iterations

        # Store ACCEPTED iterate only (default is damped unless Ng meets all criteria).
        accepted_residual = accepted - pop_old

        self._store_iteration(accepted, accepted_residual)
        self.last_iteration = iteration

        return accepted

    def _compute_change(self, pop_new: npt.NDArray[np.float64], pop_old: npt.NDArray[np.float64]) -> float:

        floor = self.config.min_relative_pop

        denom = np.maximum(np.abs(pop_old), floor)

        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(pop_new - pop_old) / denom

        return np.max(rel)

    def _store_iteration(self, pop: npt.NDArray[np.float64], residual: npt.NDArray[np.float64]) -> None:

        self.ng_history.append(pop.copy())
        self.residual_history.append(residual.copy())

        # Truncate histories
        max_hist = self.config.ng_history_length + 1

        if len(self.ng_history) > max_hist:
            self.ng_history.pop(0)

        if len(self.residual_history) > max_hist:
            self.residual_history.pop(0)

    # def update_old(
    #         self,
    #         pop_new: npt.NDArray[np.float64],
    #         pop_old: npt.NDArray[np.float64],
    #         iteration: int,
    # ) -> npt.NDArray[np.float64]:
    #     """
    #     Apply acceleration/damping for this layer.
    #
    #     Parameters:
    #         pop_new: Newly solved populations
    #         pop_old: Previous iteration populations
    #         iteration: Current iteration number (0-indexed)
    #
    #     Returns:
    #         Accelerated/damped populations
    #     """
    #
    #     # Check iteration sequence
    #     if iteration != self.last_iteration + 1 and iteration != self.last_iteration:
    #         log.warning(f"[nL{self.layer_idx}] Non-sequential iteration (I{self.last_iteration}->{iteration})")
    #
    #     # Calculate change
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         max_change = np.max(np.abs((pop_new - pop_old) / (pop_old + 1e-30)))
    #
    #     # Store change history (extend if new iteration)
    #     if iteration >= len(self.change_history):
    #         self.change_history.append(max_change)
    #     else:
    #         self.change_history[iteration] = max_change
    #
    #     # Store Ng history
    #     if iteration >= len(self.ng_history):
    #         self.ng_history.append(pop_new.copy())
    #     else:
    #         self.ng_history[iteration] = pop_new.copy()
    #
    #     if iteration < self.config.warmup_iterations:
    #         log.debug(f"[nL{self.layer_idx}] Warmup - no acceleration (max. change={max_change:.4e})")
    #         self.last_iteration = iteration
    #         return pop_new
    #
    #     # Try Ng acceleration first (if conditions met)
    #     if self._should_use_ng_old(iteration):
    #         try:
    #             pop_ng = self._apply_ng_old()
    #
    #             if self._ng_is_safe(pop_ng, pop_old):
    #                 log.info(f"[nL{self.layer_idx}] Ng acceleration (max. change={max_change:.4e})")
    #
    #                 self.last_iteration = iteration
    #                 return pop_ng
    #             else:
    #                 log.warning(f"[nL{self.layer_idx}] Ng unsafe, falling back to damping")
    #
    #                 self.ng_disabled_until_iter = iteration + self.config.ng_disable_iterations
    #
    #         except RuntimeError as e:
    #             log.warning(f"[nL{self.layer_idx}] Ng failed: {e}, using damping.")
    #             self.ng_disabled_until_iter = iteration + self.config.ng_disable_iterations
    #
    #     pop_damped = self._apply_damping(pop_new, pop_old)
    #
    #     log.debug(f"[nL{self.layer_idx}] Damping omega={self.omega:.3f} (max. change={max_change:.4e})")
    #
    #     self.last_iteration = iteration
    #     return pop_damped

    def _should_use_ng(self, iteration: int) -> bool:

        if iteration <= self.ng_disabled_until_iter:
            return False

        if len(self.ng_history) < self.config.ng_history_length:
            return False

        if len(self.change_history) < 4:
            return False

        current = self.change_history[-1]

        if current > self.config.ng_enable_threshold:
            return False

        recent = self.change_history[-4:]

        # Strict monotonic convergence
        monotonic = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))

        if not monotonic:
            return False

        # Require roughly linear convergence
        ratios = [recent[i + 1] / recent[i] for i in range(len(recent) - 1)]
        ratio_spread = max(ratios) - min(ratios)

        if ratio_spread > 0.3:
            return False

        return True

    # def _should_use_ng_old(self, iteration: int) -> bool:
    #     """Check if Ng should be attempted."""
    #     # Disabled temporarily?
    #     if iteration <= self.ng_disabled_until_iter:
    #         return False
    #
    #     # Need enough history
    #     if len(self.ng_history) < self.config.ng_history_length + 1:
    #         return False
    #
    #     # Need to be in smooth convergence regime
    #     if len(self.change_history) < 3:
    #         return False
    #
    #     current_change = self.change_history[-1]
    #     if current_change > self.config.ng_enable_threshold:
    #         return False
    #
    #     # Check for monotonic decrease (smooth convergence)
    #     recent = self.change_history[-3:]
    #     monotonic = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))
    #
    #     return monotonic

    def _apply_ng(self, pop_new: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Apply Ng acceleration in logarithmic population space using the current nonlinear iteration together with a
        history of previously accepted iterates.

        The method constructs an accelerated population vector from the most recent :math:`m+1` iterates, where
        :math:`m` = self.config.ng_history_length. The current nonlinear solution (pop_new) is temporarily appended to
        the stored history before constructing the acceleration system.

        The acceleration is performed in logarithmic population space:

        .. math::
            y_k = log(max(n_k, \\epsilon)),

        where :math:`n_k` is the population vector for iteration :math:`k`, and :math:`\\epsilon` is a small floor value
        to avoid singularities for very small populations.

        Ng differences are then constructed as:

        .. math::
            d_k = y_{k+1} - y_k.

        These differences define the Ng matrix:

        .. math::
            B_{ij} = d_i \\cdot d_j,

        where the dot product is taken over all population states. A small diagonal regularization term is added to
        improve numerical stability:

        .. math::
            B \\rightarrow B + \\lambda I.

        The Ng coefficients are obtained from:

        .. math::
            B \\alpha = 1,

        and normalized such that:

        .. math::
            \\sum_i \\alpha_i = 1.

        The accelerated logarithmic population vector is constructed by extrapolating from the current nonlinear
        solution:

        .. math::
            y_{Ng} = y_m + \\sum_i \\alpha_i d_i,

        where :math:`y_m` corresponds to the logarithm of the current nonlinear iterate.

        Finally, the accelerated populations are recovered via exponentiation and normalization:

        .. math::
            n_{Ng} = \\exp(y_{Ng}) / \\sum\\exp(y_{Ng}).

        Performing the extrapolation in logarithmic space improves robustness for NLTE population problems where state
        populations may span many orders of magnitude. The logarithmic formulation naturally preserves positivity and
        treats multiplicative population changes more uniformly than linear population-space extrapolation.

        Parameters
        ----------
        pop_new : np.ndarray
            Newly computed population vector for the current nonlinear iteration prior to damping or acceleration
            acceptance.

        Returns
        -------
        pop_ng : np.ndarray
            Ng-accelerated population vector normalized to unity.

        Raises
        -------
        RuntimeError
            Raises if any of these conditions are satisfied: the Ng matrix is ill-conditioned; extrapolation
            coefficients become excessively large; non-finite values are produced; normalization fails.
        """
        num_ng = self.config.ng_history_length + 1
        floor = self.config.min_relative_pop
        # pops = np.array(self.ng_history[-num_ng:])
        history = self.ng_history.copy()
        history.append(pop_new.copy())  # Crucial or current step info is lost.
        pops = np.array(history[-num_ng:])

        # Transform to log-space.
        log_pops = np.log(np.maximum(pops, floor))
        # Residuals in log-space
        deltas = np.diff(log_pops, axis=0)
        # deltas = deltas[-num_ng:]  # Enforced by pops slicing.
        # deltas = np.array(self.residual_history[-num_ng:])

        # Build DIIS matrix.
        mat_dim = num_ng - 1
        b_matrix = np.empty((mat_dim, mat_dim))

        for i in range(mat_dim):
            for j in range(mat_dim):
                b_matrix[i, j] = np.dot(
                    deltas[i],
                    deltas[j],
                )

        cond = np.linalg.cond(b_matrix)

        if cond > self.config.ng_max_condition:
            raise RuntimeError(f"Ill-conditioned DIIS matrix: {cond:.2e}.")

        reg = self.config.ng_regularization * np.trace(b_matrix) / mat_dim

        b_matrix += reg * np.eye(mat_dim)
        rhs = np.ones(mat_dim)

        alpha = np.linalg.solve(b_matrix, rhs)
        alpha /= alpha.sum()

        # Coefficient safety check.
        if np.any(np.abs(alpha) > self.config.ng_max_alpha):
            raise RuntimeError(f"Large DIIS coefficients: {alpha}.")

        # Extrapolate populations from alpha coefficients and differences.
        log_ng = log_pops[-1].copy()  # This is the log of pop_new.

        for a, log_dif in zip(alpha, deltas):
            log_ng += a * log_dif

        # Enforce positivity and re-normalise.
        pop_ng = np.maximum(np.exp(log_ng), 0.0)
        total = pop_ng.sum()

        if total <= 0:
            raise RuntimeError("Invalid Ng normalization.")

        pop_ng /= total
        if not np.isfinite(pop_ng).all():
            raise RuntimeError("Ng produced non-finite populations.")

        return pop_ng

    # def _apply_ng_old(self) -> npt.NDArray[np.float64]:
    #     """Apply Ng acceleration."""
    #     n = self.config.ng_history_length
    #
    #     # Extract last n+1 iterates
    #     pops = np.array(self.ng_history[-(n + 1):])
    #
    #     # Calculate differences
    #     deltas = np.diff(pops, axis=0)
    #
    #     # Build matrix A_ij = delta_n^i · delta_n^j
    #     a_matrix = np.zeros((n, n))
    #     for i in range(n):
    #         for j in range(n):
    #             a_matrix[i, j] = np.dot(deltas[i], deltas[j])
    #
    #     # Solve for weights
    #     ones = np.ones(n)
    #     reg = 1e-12 * np.trace(a_matrix)
    #     alpha = np.linalg.solve(a_matrix + reg * np.eye(n), ones)
    #     alpha /= alpha.sum()
    #
    #     # Extrapolate
    #     n_new = np.sum([alpha[i] * pops[i] for i in range(n)], axis=0)
    #
    #     if not np.isfinite(n_new).all():
    #         raise RuntimeError("Ng produced non-finite values")
    #
    #     # Clamp and normalize
    #     n_new = np.maximum(n_new, 0.0)
    #     n_new /= n_new.sum()
    #
    #     return n_new

    def _apply_damping(
            self,
            pop_new: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Apply adaptive damping. Omega is adjusted based on convergence behaviour: if populations are monotonically
        decreasing, omega tends to 1 (preference for new iteration); if populations are oscillating, omega decreases to
        slow changes, favouring populations from the previous iteration.

        Parameters
        ----------
        pop_new: np.ndarray
            Populations from new iteration.
        pop_old: np.ndarray
            Populations from previous iteration.

        Returns
        -------
            Damped populations.
        """

        if len(self.change_history) >= 3:
            recent = self.change_history[-3:]

            # Check for oscillation
            if not all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1)):
                # Non-monotonic - introduce damping.
                old_omega = self.omega
                self.omega = max(
                    self.omega * self.config.omega_decrease_factor,
                    self.config.omega_min
                )
                if self.omega < old_omega:
                    log.info(f"[nL{self.layer_idx}] Oscillating - omega={old_omega:.2f}->{self.omega:.2f}")
            else:
                # Monotonic - reduce damping (approach omega=1).
                omega_max = (
                    self.config.ng_omega_cap if self._should_use_ng(self.last_iteration) else self.config.omega_max
                )

                # Enforce cap if already above (only when Ng enabled).
                if self.omega > omega_max:
                    old_omega = self.omega
                    self.omega = omega_max

                    log.debug(f"[nL{self.layer_idx}] Reducing omega for Ng: {old_omega:.2f}->{self.omega:.2f}.")
                if self.omega < omega_max:
                    old_omega = self.omega

                    self.omega = min(
                        self.omega * self.config.omega_increase_factor,
                        omega_max
                    )
                    if self.omega > old_omega:
                        log.debug(f"[nL{self.layer_idx}] Smooth - omega={old_omega:.2f}->{self.omega:.2f}")

        pop_damped = self.omega * pop_new + (1 - self.omega) * pop_old

        if np.any(pop_damped < 0):
            log.warning(f"[nL{self.layer_idx}] Damping produced negatives, clamping.")
            pop_damped = np.maximum(pop_damped, 0.0)

        pop_damped /= pop_damped.sum()

        return pop_damped

    def _accept_ng(
            self,
            pop_ng: npt.NDArray[np.float64],
            pop_damped: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
    ) -> bool:

        if np.any(~np.isfinite(pop_ng)):
            return False

        if np.any(pop_ng < 0):
            return False

        # Compare Ng and damped step sizes.
        ng_step = np.linalg.norm(pop_ng - pop_old)
        damped_step = np.linalg.norm(pop_damped - pop_old)

        if ng_step > self.config.ng_step_limit * damped_step:
            # Reject if Ng step is larger (less convergent) than damping.
            return False

        # Reject if predicted convergence worsens.
        ng_change = self._compute_change(pop_ng, pop_old)
        damped_change = self._compute_change(pop_damped, pop_old)

        if ng_change > self.config.ng_worsening_factor * damped_change:
            return False

        return True

    def _ng_is_safe(
            self,
            pop_accel: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
    ) -> bool:
        """Check if Ng accelerated populations are safe."""
        if np.any(pop_accel < 0):
            return False
        if not np.isfinite(pop_accel).all():
            return False

        # Check jumps (optional)
        if self.config.check_max_jump:
            # Only check significant populations
            sig_mask = pop_old > self.config.min_pop_for_jump_check

            if np.any(sig_mask):
                with np.errstate(divide='ignore', invalid='ignore'):
                    jump = pop_accel[sig_mask] / pop_old[sig_mask]

                if np.any(jump > self.config.max_jump_factor):
                    return False

                if np.any(jump < 1.0 / self.config.max_jump_factor):
                    return False

        return True

    def converged(self) -> bool:
        """Check if this layer has converged."""
        if len(self.change_history) == 0:
            return False
        return self.change_history[-1] < self.config.convergence_threshold

    def get_max_change(self) -> float:
        """Get current max change for this layer."""
        if len(self.change_history) == 0:
            return np.inf
        return self.change_history[-1]


class HybridAccelerator:
    """
    Convergence acceleration/damping manager for separate layers of a given species.
    """

    __slots__ = ["n_layers", "config", "layer_accelerators"]

    def __init__(
            self,
            n_layers: int,
            config: AccelerationConfig = None,
    ):
        """

        Parameters
        ----------
        n_layers: int
            Number of layers to track; should be the number of non-LTE layers for a given species.
        config: dict
            Acceleration configuration.
        """
        if config is None:
            config = AccelerationConfig()

        self.n_layers = n_layers
        self.config = config
        self.layer_accelerators = [
            LayerAccelerator(layer_idx=layer_idx, config=config)
            for layer_idx in range(n_layers)
        ]

    def update(
            self,
            pop_new: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
            iteration: int,
            layer_idx: int,
    ) -> npt.NDArray[np.float64]:
        """
        Apply acceleration for a specific layer. Populations should be normalised to 1 for consistency; scale by partial
        normlisation factor outside of this utility.

        Parameters
        ----------
        pop_new: np.ndarray
            Populations from new iteration.
        pop_old: np.ndarray
            Populations from previous iteration.
        iteration: int
            Current iteration number.
        layer_idx: int
            Layer index.

        Returns
        -------
            Accelerated/damped populations.
        """
        return self.layer_accelerators[layer_idx].update(
            pop_new=pop_new,
            pop_old=pop_old,
            iteration=iteration,
        )

    def converged(self, layer_idx: int = None) -> bool:
        """

        Parameters
        ----------
        layer_idx: int
            If provided, checks specific layer. If None, checks if all layers have converged.

        Returns
        -------
            True if converged.
        """
        if layer_idx is not None:
            return self.layer_accelerators[layer_idx].converged()
        else:
            return all(acc.converged() for acc in self.layer_accelerators)

    def get_max_change(self, layer_idx: int = None) -> float:
        """

        Parameters
        ----------
        layer_idx: int
            If provided, get for specific layer. If None, get maximum across all layers.

        Returns
        -------
            Maximum population change.
        """
        if layer_idx is not None:
            return self.layer_accelerators[layer_idx].get_max_change()
        else:
            return max(
                acc.get_max_change()
                for acc in self.layer_accelerators
            )

    def get_max_changes(self) -> npt.NDArray[np.float64]:
        """

        Returns
        -------
            Array containing the maximum population changes across all layers.
        """
        return np.array([acc.get_max_change() for acc in self.layer_accelerators])
