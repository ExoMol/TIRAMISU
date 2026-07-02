import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import logging

from .config import _LOG_VERBOSE_1, _LOG_VERBOSE_2

log = logging.getLogger(__name__)


@dataclass
class AccelerationConfig:
    """Configuration for acceleration methods."""
    # Startup behaviour.
    warmup_iterations: int = 3  # No acceleration for first N iterations.
    # For Ng, this can only kick in after warmup_iterations + ng_history_length iterations.

    # Adaptive damping
    omega_start: float = 1.0
    omega_min: float = 0.25
    omega_max: float = 1.0
    omega_increase_factor: float = 1.05  # Recovering from oscillation; return preference to new step.
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
        # Raw change stores new raw pops - previous accepted.
        self.change_history = []  # List of floats, one per iteration
        # register_raw initially stores the raw new pops to compare against previous iterations; but once apply is
        # called this is changed to the accepted value.

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
        Single-species update: equivalent to register_raw() followed immediately by apply().
        Preserved for backward compatibility with code that uses HybridAccelerator directly.
        For multi-species use, call register_raw() across all species first, then apply().
        """
        self.register_raw(pop_new, pop_old, iteration)
        return self.apply(pop_new, pop_old, iteration)

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

        # Use the median of recent changes rather than the last single value so that one anomalously low iterate (e.g.
        # a chance near-cancellation) does not prematurely gate Ng in an otherwise stagnant sequence.
        recent = self.change_history[-4:]
        # median_change = float(np.median(recent))
        #
        # if median_change > self.config.ng_enable_threshold:
        #     return False

        # Strict monotonic convergence
        monotonic = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))

        if not monotonic:
            return False

        # Accept a general downward trend, rather than strict per-step monotonicity. For oscillating species the raw
        # per-layer change bounces even when the sequence as a whole is converging, so strict monotonicity almost never
        # fires. A positive Theil-Sen slope (median pairwise slopes) is sufficient.
        # pairs = [(recent[j + 1] - recent[j]) for j in range(len(recent) - 1)]
        # trend = float(np.median(pairs))  # negative → improving on average
        #
        # if trend > 0:
        #     # Diverging.
        #     return False

        # # Require roughly linear convergence
        ratios = [recent[i + 1] / recent[i] for i in range(len(recent) - 1)]
        ratio_spread = max(ratios) - min(ratios)

        if ratio_spread > 0.3:
            return False

        return True

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

        # Build the working history from stored iterates plus the current raw pop_new.
        # Use the stored ng_history (accepted/damped iterates) for all historical points but replace the last slot
        # conceptually by appending pop_new, so that the deltas span the genuine nonlinear residuals rather than the
        # compressed damped steps. When omega << 1 the damped history occupies a very narrow band in log-space and the
        # resulting deltas become nearly collinear, causing the B matrix to be ill-conditioned.
        history = self.ng_history.copy()
        history.append(pop_new.copy())  # Crucial or current step info is lost.
        pops = np.array(history[-num_ng:])

        # Transform to log-space.
        log_pops = np.log(np.maximum(pops, floor))
        # Residuals in log-space
        deltas = np.diff(log_pops, axis=0)

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

        # Regularisation: relative to trace with a hard floor to handle near-zero traces.
        trace = np.trace(b_matrix)
        reg_scale = max(abs(trace) / mat_dim, 1e-30)
        reg = self.config.ng_regularization * reg_scale
        # reg = self.config.ng_regularization * np.trace(b_matrix) / mat_dim

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

    def register_raw(
            self,
            pop_new: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
            iteration: int,
    ) -> None:
        """
        Phase 1 of the two-phase species-aware update.

        Record the raw (pre-acceleration) change for this layer without yet committing an accepted iterate. Called by
        MultiSpeciesAccelerator for every species on every layer before any damping decision is made, so that
        cross-species oscillation detection can aggregate signals across all species before applying damping.

        Parameters
        ----------
        pop_new : ndarray
            Newly solved populations for this layer (normalised to 1).
        pop_old : ndarray
            Previous-iteration populations for this layer (normalised to 1).
        iteration : int
            Current iteration number (0-indexed).
        """
        max_change = self._compute_change(pop_new, pop_old)

        self.change_history.append(max_change)

        # if iteration >= len(self.change_history):
        #     self.change_history.append(max_change)
        # else:
        #     self.change_history[iteration] = max_change

    def apply(
            self,
            pop_new: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
            iteration: int,
            omega_floor: float = None,
    ) -> npt.NDArray[np.float64]:
        """
        Phase 2 of the two-phase species-aware update.

        Apply damping and (if eligible) Ng acceleration, using the change already recorded by ``register_raw``. The
        ``omega_floor`` argument allows MultiSpeciesAccelerator to impose a global damping floor when oscillation is
        detected in any species on this layer, ensuring all species are damped consistently even if an individual
        species appears locally smooth.

        Parameters
        ----------
        pop_new : ndarray
            Newly solved populations (normalised to 1).
        pop_old : ndarray
            Previous-iteration populations (normalised to 1).
        iteration : int
            Current iteration number (0-indexed).
        omega_floor : float or None
            If provided, caps ``self.omega`` from below during this iteration. Does not permanently alter the stored
            omega; the per-layer adaptive logic continues normally so that omega can recover once oscillation subsides.

        Returns
        -------
        ndarray
            Accepted (accelerated/damped) populations.
        """
        residual = pop_new - pop_old

        # Warmup: accept raw iterate, store history, return immediately.
        if iteration < self.config.warmup_iterations:
            accepted = pop_new.copy()
            self._store_iteration(accepted, residual)
            self.last_iteration = iteration
            return accepted

        # Compute damped baseline, temporarily honouring the cross-species floor if supplied.
        pop_damped = self._apply_damping(pop_new, pop_old, omega_floor=omega_floor)
        accepted = pop_damped

        # Attempt Ng/DIIS if eligible.
        if self._should_use_ng(iteration):
            try:
                pop_ng_raw = self._apply_ng(pop_new)
                pop_ng = (
                        self.config.ng_mix * pop_ng_raw
                        + (1.0 - self.config.ng_mix) * pop_damped
                )
                pop_ng /= pop_ng.sum()

                if self._accept_ng(pop_ng=pop_ng, pop_damped=pop_damped, pop_old=pop_old):
                    accepted = pop_ng
                    log.log(_LOG_VERBOSE_2, f"[nL{self.layer_idx}] Accepted Ng acceleration.")
                else:
                    log.log(_LOG_VERBOSE_2, f"[nL{self.layer_idx}] Rejected Ng acceleration.")
                    self.ng_disabled_until_iter = iteration + self.config.ng_disable_iterations

            except RuntimeError as e:
                log.warning(f"[nL{self.layer_idx}] Ng failed: {e}.")
                self.ng_disabled_until_iter = iteration + self.config.ng_disable_iterations

        accepted_residual = accepted - pop_old
        self._store_iteration(accepted, accepted_residual)
        self.last_iteration = iteration

        # Crucial step: if this is not updated, only the raw changes are stored and the convergence checks never see the
        # actual accepted changes per layer after damping/Ng is applied! So the accepted pops stored might be converged,
        # but the accelerator would be checking differences in pops that were never accepted.
        accepted_change = self._compute_change(accepted, pop_old)
        self.change_history[-1] = accepted_change

        return accepted

    def _apply_damping(
            self,
            pop_new: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
            omega_floor: float = None,
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
        omega_floor : float or None
            Cross-species damping floor for this iteration. When supplied, the effective omega used for mixing is
            ``max(self.omega, omega_floor)`` inverted — i.e. ``min(self.omega, omega_floor_as_upper_cap)``. Concretely:
            if any other species is oscillating on this layer, omega_floor will be low (e.g. 0.5) and this species will
            also be damped to at most that value for this iteration, even if its own history looks monotone. The stored
            ``self.omega`` is not permanently modified; adaptive logic continues normally.

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
                    log.log(_LOG_VERBOSE_2,
                            f"[nL{self.layer_idx}] Oscillating - omega={old_omega:.2f}->{self.omega:.2f}")
            else:
                # Monotonic - reduce damping (approach omega=1).
                omega_max = (
                    self.config.ng_omega_cap if self._should_use_ng(self.last_iteration) else self.config.omega_max
                )

                # Enforce cap if already above (only when Ng enabled).
                if self.omega > omega_max:
                    old_omega = self.omega
                    self.omega = omega_max

                    log.info(
                        _LOG_VERBOSE_2,
                        f"[nL{self.layer_idx}] Reducing omega for Ng: {old_omega:.2f}->{self.omega:.2f}."
                    )
                if self.omega < omega_max:
                    old_omega = self.omega

                    self.omega = min(
                        self.omega * self.config.omega_increase_factor,
                        omega_max
                    )
                    if self.omega > old_omega:
                        log.log(
                            _LOG_VERBOSE_2,
                            f"[nL{self.layer_idx}] Smooth - omega={old_omega:.2f}->{self.omega:.2f}"
                        )

        # Apply cross-species damping floor for this iteration only; does not alter self.omega.
        effective_omega = self.omega
        if omega_floor is not None and omega_floor < effective_omega:
            log.log(
                _LOG_VERBOSE_2,
                f"[nL{self.layer_idx}] Cross-species omega floor applied: "
                f"{effective_omega:.2f}->{omega_floor:.2f}"
            )
            effective_omega = omega_floor

        # pop_damped = self.omega * pop_new + (1 - self.omega) * pop_old
        pop_damped = effective_omega * pop_new + (1 - effective_omega) * pop_old

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

        # Reject if the Ng iterate is further from pop_old in relative terms than the damped iterate.
        ng_change = self._compute_change(pop_ng, pop_old)
        damped_change = self._compute_change(pop_damped, pop_old)
        # raw_change = self.change_history[-1]

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
        """
        Check if this layer has converged.

        Convergence is obtained if:
            - The most recent maximum population change is less than the convergence threshold.
            - The 4 most recent maximum population changes show minimal fluctuations about their mean, with the mean
             being less than twice the convergence threshold.

        The latter criterion is allowed for cases where populations oscillate between two solutions. This likely occurs
        due to the coupling between species oscillating population s back and forth in a manner than cannot be damped
        below the scale of the oscillations, i.e.: the changes in one species trigger changes in another, which revert
        the first species' changes, and these alterations overshoot a stable minima between them.
        """
        if len(self.change_history) == 0:
            return False
        # return self.change_history[-1] < self.config.convergence_threshold
        if self.change_history[-1] < self.config.convergence_threshold:
            return True
        recent = self.change_history[-4:]
        tight_changes = max(recent) - min(recent) < 0.1 * np.mean(recent)
        close_mean_changes = np.mean(recent) < 2.0 * self.config.convergence_threshold
        return tight_changes and close_mean_changes

    def get_max_change(self) -> float:
        """Get current max change for this layer."""
        if len(self.change_history) == 0:
            return np.inf
        return self.change_history[-1]


class MultiSpeciesAccelerator:
    """
    Species-aware convergence accelerator for coupled NLTE population systems.

    Manages a :class:`tiramisu.accelerator.LayerAccelerator` for every (species, layer) pair and implements a two-phase
    update protocol that enables cross-species damping:

    **Step 1, :func:`tiramisu.accelerator.MultiSpeciesAccelerator.register_raw`
        Call once per species per layer, passing the raw new and old populations. Each LayerAccelerator records the
        per-layer maximum relative change for this iteration without yet committing an accepted iterate. After all
        species have been registered for a given layer, the global oscillation signal for that layer is determined by
        inspecting whether any species is oscillating.

    **Step 2, :func:`tiramisu.accelerator.MultiSpeciesAccelerator.apply_all`
        Call once all species have been registered for all layers. Applies damping and Ng acceleration to every
        (species, layer), passing a per-layer ``omega_floor`` to each LayerAccelerator when cross-species oscillation
        was detected on that layer. Each species' own adaptive omega logic continues independently; the floor only ever
        tightens damping for the current iteration and does not permanently alter the stored omega of a locally smooth
        species.

    The caller is responsible for supplying ``pop_new`` and ``pop_old`` consistently between the
    two phases. The typical calling pattern mirrors the existing workflow structure:

    .. code-block:: python

        # --- Phase 1: collect raw updates for all species on all NLTE layers ---
        raw_pops: dict[str, dict[int, tuple]] = {}
        for species, processor in nlte_processors.items():
            raw_pops[species] = {}
            for layer_idx in nlte_layer_indices:
                pop_new, pop_old = processor.solve_pops_raw(layer_idx, ...)
                multi_accel.register_raw(species, layer_idx, pop_new, pop_old, n_iter)
                raw_pops[species][layer_idx] = (pop_new, pop_old)

        # --- Phase 2: apply acceleration with cross-species damping floors ---
        for species, processor in nlte_processors.items():
            for layer_idx in nlte_layer_indices:
                pop_new, pop_old = raw_pops[species][layer_idx]
                accepted = multi_accel.apply(species, layer_idx, pop_new, pop_old, n_iter)
                processor.finalise_layer(layer_idx, accepted, ...)

    Parameters
    ----------
    species_layers : dict[str, int]
        Mapping from species name to the number of NLTE layers for that species.  All species must have the same layer
        count in the current implementation (since they share the same atmospheric layer grid), but the interface
        accepts per-species counts for forward compatibility.
    config : AccelerationConfig or None
        Shared acceleration configuration. If None, defaults are used.
    """

    __slots__ = ["species_layers", "config", "accelerators", "_oscillating_layers"]

    def __init__(
            self,
            species_layers: dict[str, int],
            config: AccelerationConfig = None,
    ):
        if config is None:
            config = AccelerationConfig()

        self.config = config
        self.species_layers = species_layers

        # One LayerAccelerator per (species, layer).
        self.accelerators: dict[str, list[LayerAccelerator]] = {
            species: [
                LayerAccelerator(layer_idx=layer_idx, config=config)
                for layer_idx in range(n_layers)
            ]
            for species, n_layers in species_layers.items()
        }

        # Per-layer oscillation flag, reset each iteration by register_raw.
        # Key: layer_idx (NLTE-relative). Value: bool; True if any species oscillating.
        self._oscillating_layers: dict[int, bool] = {}

    def register_raw(
            self,
            species: str,
            layer_idx: int,
            pop_new: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
            iteration: int,
    ) -> None:
        """
        Record the raw population change for one species on one layer.

        Must be called for every species on every layer before ``apply`` is called for any of them. After all species
        have been registered for a given layer, the cross-species oscillation signal for that layer is updated
        automatically.

        Parameters
        ----------
        species : str
            Species name (must be a key in ``species_layers``).
        layer_idx : int
            NLTE-relative layer index (i.e.: ``layer_idx - n_lte_layers``).
        pop_new : ndarray
            Newly solved populations, normalised to 1.
        pop_old : ndarray
            Previous-iteration populations, normalised to 1.
        iteration : int
            Current iteration number (0-indexed).
        """
        la = self.accelerators[species][layer_idx]
        la.register_raw(pop_new, pop_old, iteration)

        # After registering, check whether this species is oscillating on this layer.
        # Requires at least 3 entries in change_history.
        if layer_idx not in self._oscillating_layers:
            self._oscillating_layers[layer_idx] = False

        if len(la.change_history) >= 3:
            recent = la.change_history[-3:]
            # Oscillation is assumed if not monotonic.
            is_oscillating = not all(recent[i] >= recent[i + 1] for i in range(len(recent) - 1))
            # Once any species is oscillating on this layer, the floor applies to all.
            if is_oscillating:
                self._oscillating_layers[layer_idx] = True
                log.log(
                    _LOG_VERBOSE_2,
                    f"[nL{layer_idx}] Cross-species oscillation detected from {species}; damping floor applied."
                )

    def apply(
            self,
            species: str,
            layer_idx: int,
            pop_new: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
            iteration: int,
    ) -> npt.NDArray[np.float64]:
        """
        Apply damping and Ng acceleration for one species on one layer.

        Must be called only after ``register_raw`` has been called for all species on this layer in the current
        iteration. The cross-species omega floor is determined by the aggregated oscillation signal set during the
        register_raw logic.

        Parameters
        ----------
        species : str
            Species name.
        layer_idx : int
            NLTE-relative layer index.
        pop_new : ndarray
            Same array that was passed to ``register_raw`` for this (species, layer, iteration).
        pop_old : ndarray
            Same array that was passed to ``register_raw``.
        iteration : int
            Current iteration number (0-indexed).

        Returns
        -------
        ndarray
            Accepted (accelerated/damped) populations, normalised to 1.
        """
        omega_floor = self.config.omega_min if self._oscillating_layers.get(layer_idx, False) else None
        return self.accelerators[species][layer_idx].apply(
            pop_new=pop_new,
            pop_old=pop_old,
            iteration=iteration,
            omega_floor=omega_floor,
        )

    def reset_oscillation_flags(self) -> None:
        """
        Clear per-layer oscillation flags at the start of each iteration.

        Must be called once per iteration, before any ``register_raw`` calls, so that the flags reflect only the current
        iteration's signals and do not carry over stale state.
        """
        self._oscillating_layers = {}

    def converged(self, species: str = None, layer_idx: int = None) -> bool:
        """
        Check convergence.

        - ``species=None, layer_idx=None``: all species, all layers.
        - ``species=X, layer_idx=None``: all layers for species X.
        - ``species=X, layer_idx=N``: specific (species, layer).
        """
        if species is not None and layer_idx is not None:
            return self.accelerators[species][layer_idx].converged()
        elif species is not None:
            return all(la.converged() for la in self.accelerators[species])
        else:
            return all(
                la.converged()
                for las in self.accelerators.values()
                for la in las
            )

    def get_max_change(self, species: str = None, layer_idx: int = None) -> float:
        """
        Return the maximum relative population change.

        - ``species=None, layer_idx=None``: global maximum across all species and layers.
        - ``species=X, layer_idx=None``: maximum across all layers for species X.
        - ``species=X, layer_idx=N``: specific (species, layer).
        """
        if species is not None and layer_idx is not None:
            return self.accelerators[species][layer_idx].get_max_change()
        elif species is not None:
            return max(la.get_max_change() for la in self.accelerators[species])
        else:
            return max(
                la.get_max_change()
                for las in self.accelerators.values()
                for la in las
            )

    def get_max_changes(self, species: str) -> npt.NDArray[np.float64]:
        """Return per-layer maximum changes for a given species."""
        return np.array([la.get_max_change() for la in self.accelerators[species]])
