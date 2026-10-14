import typing as t
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import logging

from .config import _LOG_VERBOSE_2

log = logging.getLogger(__name__)


@dataclass
class AccelerationConfig:
    """Configuration for acceleration methods."""
    # Startup behaviour.
    warmup_iterations: int = 3  # No acceleration for first N iterations.

    # Adaptive damping
    omega_start: float = 1.0
    omega_min: float = 0.5
    omega_max: float = 1.0
    omega_decrease_factor: float = 0.7  # Oscillating; prefer pops from previous iteration.

    # Anderson extrapolation. The number of residual vectors actually used each call is determined adaptively via
    # SVD truncation (`svd_rcond`) from a candidate window of anderson_window + 1, not fixed in advance - see
    # _solve_extrapolation_coefficients.
    method: str = "anderson"  # "anderson", or "damping_only" (no extrapolation). Shared across every species/layer -
    # there is no per-species override.
    extrapolation_history_size: int = 15  # Rolling buffer of past iterates/residuals retained. Must be >=
    # anderson_window + 1.
    anderson_window: int = 8  # Anderson depth: the last anderson_window + 1 (input, output) pairs. This is an upper
    # bound on the candidate pool, not a fixed count - see _solve_extrapolation_coefficients for how the effective
    # window is chosen adaptively each call.
    svd_rcond: float = 1e-8  # Relative singular-value cutoff (vs. the largest) used to adaptively truncate the
    # extrapolation subspace to its well-conditioned part.
    extrapolation_ridge: float = 1e-8  # Small additional Tikhonov ridge, scaled to the *largest retained* singular
    # value squared (not the trace/mean - see _solve_extrapolation_coefficients docstring for why that matters).

    # Extrapolation safety envelope
    extrapolation_disable_iterations: int = 5  # Disable for N iters after failure.
    anderson_max_alpha: float = 50.0  # Coefficient-magnitude guard. Permissive: legitimately well-conditioned
    # solutions naturally need larger, sign-alternating coefficients to cancel a many-candidate, partially-redundant
    # history. The physically-grounded step-size/change checks in _accept_extrapolation remain the real safety net.
    extrapolation_step_limit: float = 2.0
    extrapolation_worsening_factor: float = 2.0  # 1.2
    min_relative_pop: float = 1e-70
    extrapolation_mix: float = 1.0  # 0.5
    extrapolation_omega_cap = 0.8
    extrapolation_patience: int = 3  # Restart a layer's history when, right after an accepted extrapolation, the last
    # this-many requested changes are all above the smallest one before them since the history started (see
    # MultiSpeciesAccelerator.apply).

    # Convergence
    convergence_threshold: float = 1e-3  # 0.1%.


def _adapt_omega(
        omega: float,
        oscillating: bool,
        config: "AccelerationConfig",
        extrapolation_active: bool,
) -> float:
    """
    Pure update rule for the adaptive damping factor omega.

    Encodes the damping rules: if the solve is oscillating (successive requested steps point in opposite directions,
    see :meth:`MultiSpeciesAccelerator._update_layer_omega`), omega is decreased (bounded below by
    ``config.omega_min``) to favour the previous iterate more strongly. Otherwise, omega is relaxed back towards 1
    (bounded above by ``config.omega_max``, or by the tighter ``config.extrapolation_omega_cap`` when extrapolation is
    active, since it needs a sufficiently compressed damped history to remain well-conditioned).

    Damping only helps an overshooting mode: with a damping factor omega, a mode that multiplies the requested step by
    lambda each iteration is multiplied by 1 - omega (1 - lambda) instead. That brings lambda < 0 (overshoot, the
    step flips sign) towards 0, but pushes 0 < lambda < 1 (a slow drift, the step keeps its sign) towards 1, slowing it.

    This function is stateless: it takes the current omega and returns the next value. Used by
    :class:`MultiSpeciesAccelerator` for the single omega shared across all species on a given atmospheric layer.

    Parameters
    ----------
    omega : float
        Current value of the damping factor.
    oscillating : bool
        Whether the last two requested steps on the layer point in opposite directions.
    config : AccelerationConfig
        Shared acceleration configuration.
    extrapolation_active : bool
        Whether Anderson extrapolation is eligible/active for the relevant accelerator(s) this iteration. Used only
        to select the upper cap on omega when not oscillating.

    Returns
    -------
    float
        The updated omega value.
    """
    if oscillating:
        return max(omega * config.omega_decrease_factor, config.omega_min)

    omega_max = config.extrapolation_omega_cap if extrapolation_active else config.omega_max

    if omega > omega_max:
        return omega_max

    if omega < omega_max:
        return min(omega * (1.1 + 0.8 * (1 - omega)), omega_max)

    return omega


def _solve_extrapolation_coefficients(
        residuals: npt.NDArray[np.float64],
        max_alpha: float,
        svd_rcond: float,
        extrapolation_ridge: float,
        layer_idx: int,
) -> npt.NDArray[np.float64]:
    """
    Solve the DIIS/Anderson-type constrained least-squares problem

    .. math::
        \\alpha = \\arg\\min_a \\|D a\\|^2 \\quad \\text{subject to} \\quad \\sum_i a_i = 1,

    where the columns of :math:`D` are the supplied residual vectors ``residuals`` (one row per residual).

    This is solved via the (economy) SVD of :math:`D` directly, rather than by explicitly forming the Gram
    matrix :math:`B = D^T D` (as the previous implementation did) and adding a scalar Tikhonov term relative to
    its trace. That approach has two compounding problems:

    1. Forming :math:`B` squares the condition number of the underlying problem
       (:math:`\\mathrm{cond}(B) = \\mathrm{cond}(D)^2`), and can wash out genuinely small but meaningful
       singular values in floating point - the standard reason explicit normal-equations solves are avoided
       in numerical linear algebra whenever the alternative is available.
    2. Regularizing relative to :math:`\\mathrm{trace}(B) / m` (the *mean* eigenvalue) rather than the *largest*
       eigenvalue means the reference scale can be dominated by a few large eigenvalues while the small ones
       actually causing the ill-conditioning sit many orders of magnitude below it - so the regularization
       barely touches the modes it needs to.

    Working from the SVD directly also gives a principled way to size the extrapolation window *adaptively*:
    rather than trying to pick the "right" number of history vectors in advance, this method is handed a
    generous candidate pool (``residuals`` can have as many rows as ``config.anderson_window + 1``)
    and retains only the singular values that are numerically significant relative to the largest one
    (``singular_value > config.svd_rcond * largest_singular_value``). Any candidate vectors beyond the problem's
    true rank - which is bounded by the number of independent population directions, i.e. at most
    ``n_states - 1`` given the sum-to-one constraint, and will typically be far smaller than a large candidate
    window like ``anderson_window`` - are automatically discarded rather than forced into an ill-conditioned
    solve. A small additional ridge, scaled to the *largest retained* singular value squared, is applied on top
    of the truncation for extra smoothing.

    Parameters
    ----------
    residuals : ndarray, shape (m, n_states)
        Candidate residual vectors, one per row, most recent last.
    max_alpha : float
        Coefficient-magnitude guard (``config.anderson_max_alpha``): a large window structurally needs larger
        cancelling coefficients even when well-conditioned.
    svd_rcond : float
    extrapolation_ridge : float
    layer_idx : int

    Returns
    -------
    alpha : ndarray, shape (m,)
        Extrapolation coefficients, normalised to sum to 1.

    Raises
    ------
    RuntimeError
        If the leading singular value is zero/non-finite, if no singular values survive truncation, if the
        resulting coefficients are non-finite or unnormalisable, or if the coefficients are excessively large.
    """
    # D: n_states x m, columns are the residual vectors.
    d_matrix = residuals.T
    num_candidates = d_matrix.shape[1]

    # Economy SVD - k = min(n_states, m) singular values/vectors.
    u, s, vt = np.linalg.svd(d_matrix, full_matrices=False)

    if s[0] <= 0 or not np.isfinite(s[0]):
        raise RuntimeError("Degenerate residual matrix (zero or non-finite leading singular value).")

    mask = s > svd_rcond * s[0]

    if not np.any(mask):
        raise RuntimeError("No singular values survived truncation - residual matrix is numerically null.")

    s_trunc = s[mask]
    v_trunc = vt[mask, :].T  # m x k'

    if log.isEnabledFor(_LOG_VERBOSE_2):
        log.log(
            _LOG_VERBOSE_2,
            f"[nL{layer_idx}] Extrapolation using {mask.sum()}/{num_candidates} candidate vectors "
            f"(rank-truncated at svd_rcond={svd_rcond:.1e})."
        )

    ridge = extrapolation_ridge * s_trunc[0] ** 2
    inv_eigs = 1.0 / (s_trunc ** 2 + ridge)

    ones = np.ones(num_candidates)
    alpha = v_trunc @ (inv_eigs * (v_trunc.T @ ones))

    alpha_sum = alpha.sum()

    if not np.isfinite(alpha).all() or abs(alpha_sum) < 1e-300:
        raise RuntimeError("Extrapolation coefficients are non-finite or unnormalisable.")

    alpha /= alpha_sum

    if np.any(np.abs(alpha) > max_alpha):
        raise RuntimeError(f"Large extrapolation coefficients: {alpha}.")

    return alpha


def _apply_extrapolation(
        raw_history: t.List[npt.NDArray[np.float64]],
        input_history: t.List[npt.NDArray[np.float64]],
        window: int,
        max_alpha: float,
        floor: float,
        svd_rcond: float,
        ridge: float,
        layer_idx: int,
) -> npt.NDArray[np.float64]:
    """
    Anderson (DIIS, type II) extrapolation in logarithmic population space from the most recent ``window + 1``
    pairs of solver input :math:`x_j` (the accepted populations the rates were built from) and raw solver output
    :math:`g_j = G(x_j)`.

    With :math:`y = \\log(\\max(n, \\epsilon))` (:math:`\\epsilon` a small floor against vanishing populations), the
    residual of each pair is :math:`f_j = y(g_j) - y(x_j)`. The coefficients :math:`\\alpha` minimise
    :math:`\\|\\sum_j \\alpha_j f_j\\|` subject to :math:`\\sum_j \\alpha_j = 1` (see
    :func:`_solve_extrapolation_coefficients`), and the extrapolated iterate is

    .. math::
        y_{\\mathrm{ext}} = \\sum_j \\alpha_j \\, y(g_j).

    Each residual belongs to its own input, so the history stays valid whatever damping or extrapolation produced
    each :math:`x_j`. Working in log space keeps the result positive and treats populations spanning many orders of
    magnitude evenly.

    Parameters
    ----------
    raw_history, input_history : list of ndarray
        Raw solver outputs :math:`g_j` and their inputs :math:`x_j`, one per iteration, most recent last.
    window : int
        Anderson depth: ``window + 1`` pairs are combined.
    max_alpha : float
    floor : float
    svd_rcond : float
    ridge : float
    layer_idx : int

    Returns
    -------
    pop_ext : np.ndarray
        Extrapolated population vector, left unnormalised: callers building a multi-species vector (see
        :class:`MultiSpeciesAccelerator`) renormalise each species' segment independently, since the concatenation is
        not itself a single probability distribution - see :func:`_renormalize_segments`.

    Raises
    ------
    RuntimeError
        Propagated from :func:`_solve_extrapolation_coefficients`, or raised if the result is non-finite.
    """
    num_hist = window + 1
    log_g = np.log(np.maximum(np.array(raw_history[-num_hist:]), floor))
    log_x = np.log(np.maximum(np.array(input_history[-num_hist:]), floor))

    alpha = _solve_extrapolation_coefficients(
        residuals=log_g - log_x, max_alpha=max_alpha, svd_rcond=svd_rcond, extrapolation_ridge=ridge,
        layer_idx=layer_idx,
    )

    pop_ext = np.exp(alpha @ log_g)
    if not np.isfinite(pop_ext).all():
        raise RuntimeError("Extrapolation produced non-finite populations.")

    return pop_ext


def _renormalize_segments(vec: npt.NDArray[np.float64], sizes: t.List[int]) -> npt.NDArray[np.float64]:
    """
    Renormalise each contiguous per-species segment of a concatenated population vector to sum to 1.

    ``vec`` is a concatenation (in a fixed species order) of several independent population distributions - one per
    species - stacked together only so that a single Anderson extrapolation step can be fit jointly across all of
    them (see :class:`MultiSpeciesAccelerator`). Each segment must still individually sum to 1 to remain a valid set
    of fractional level populations, so normalisation has to happen per segment rather than once over the whole
    vector.

    Parameters
    ----------
    vec : ndarray
        Concatenated population vector.
    sizes : list[int]
        Length of each species' segment, in the same order used to build ``vec``.

    Returns
    -------
    ndarray
        Copy of ``vec`` with every segment normalised to sum to 1.

    Raises
    ------
    RuntimeError
        If any segment sums to a non-finite or non-positive value.
    """
    out = vec.copy()
    offset = 0
    for size in sizes:
        seg = out[offset:offset + size]
        total = seg.sum()
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("Invalid extrapolation normalization for a species segment.")
        seg /= total
        offset += size
    return out


def _compute_max_change(
        pop_new: npt.NDArray[np.float64], pop_old: npt.NDArray[np.float64], floor: float
) -> float:
    """Maximum relative population change between two iterates, floored to avoid division blow-up near zero."""
    denom = np.maximum(np.abs(pop_old), floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(pop_new - pop_old) / denom
    return np.max(rel)


def _accept_extrapolation(
        pop_accel: npt.NDArray[np.float64],
        pop_damped: npt.NDArray[np.float64],
        pop_old: npt.NDArray[np.float64],
        config: "AccelerationConfig",
        layer_idx: int,
) -> bool:
    """Shared accept/reject safety envelope for an extrapolated step, vs. its damped baseline."""
    if np.any(~np.isfinite(pop_accel)):
        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(_LOG_VERBOSE_2, "Rejected extrapolation due to infinite populations.")
        return False

    if np.any(pop_accel < 0):
        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(_LOG_VERBOSE_2, "Rejected extrapolation due to negative populations.")
        return False

    accel_step = np.linalg.norm(pop_accel - pop_old)
    damped_step = np.linalg.norm(pop_damped - pop_old)

    if accel_step > config.extrapolation_step_limit * damped_step:
        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(
                _LOG_VERBOSE_2,
                f"[nL{layer_idx}] Rejected extrapolation as step 2-norm more than {config.extrapolation_step_limit}"
                f" times damped."
            )
        return False

    accel_change = _compute_max_change(pop_accel, pop_old, config.min_relative_pop)
    damped_change = _compute_max_change(pop_damped, pop_old, config.min_relative_pop)

    if accel_change > config.extrapolation_worsening_factor * damped_change:
        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(
                _LOG_VERBOSE_2,
                f"[nL{layer_idx}] Rejected extrapolation as change greater than {config.extrapolation_worsening_factor}"
                f" times damped change."
            )
        return False

    return True


class LayerAccelerator:
    """
    Per-(species, layer) bookkeeping and damping.

    Tracks max-change/accepted-population history for one species on one atmospheric layer and applies damping to
    it. Anderson extrapolation is no longer performed here: species on the same layer are radiatively coupled, so
    it is fit jointly across all of them by :class:`MultiSpeciesAccelerator` instead (see its ``apply``) - a single
    species' isolated history is not enough context to extrapolate against.
    """

    __slots__ = ["layer_idx", "config", "max_change_history", "skipped"]

    def __init__(self, layer_idx: int, config: AccelerationConfig):
        self.layer_idx = layer_idx
        self.config = config
        # Skipped is False, unless marked as True during runtime when the species' VMR is below a VMR threshold.
        self.skipped = False

        # History storage - grows dynamically as iterations proceed. One float per iteration: the maximum relative
        # change the solve requests (raw output vs input), not the accepted step - see MultiSpeciesAccelerator.apply.
        self.max_change_history: t.List[float] = []

    def _apply_damping(
            self,
            pop_new: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
            omega: float,
    ) -> npt.NDArray[np.float64]:
        """
        Apply damping with an already-decided omega.

        ``omega`` is always supplied by :class:`MultiSpeciesAccelerator`, which computes a single damping factor per
        atmospheric layer, shared across every species on that layer, so that a full step for one species and a
        damped step for another cannot leave the combined per-layer population vector off the physical trajectory.

        Returns
        -------
            Damped populations.
        """
        pop_damped = omega * pop_new + (1 - omega) * pop_old

        if np.any(pop_damped < 0):
            log.warning(f"[nL{self.layer_idx}] Damping produced negatives, clamping.")
            pop_damped = np.maximum(pop_damped, 0.0)

        pop_damped /= pop_damped.sum()

        return pop_damped

    def converged(self) -> bool:
        """
        Check if this layer has converged.

        The changes are those the solve requests (raw output vs input), so a layer only converges when its populations
        satisfy their own rate equations, whatever damping or extrapolation produced them.

        Convergence is obtained if:
            - The most recent maximum population change is less than the convergence threshold.
            - The 4 most recent maximum population changes show minimal fluctuations about their mean, with the mean
             being less than twice the convergence threshold.

        The latter criterion is allowed for cases where populations oscillate between two solutions. This likely occurs
        due to the coupling between species oscillating population s back and forth in a manner than cannot be damped
        below the scale of the oscillations, i.e.: the changes in one species trigger changes in another, which revert
        the first species' changes, and these alterations overshoot a stable minima between them.

        Additionally, the layer is treated as converged if it is skipped due to negligible VMR.
        """
        if self.skipped:
            # Layer ignored due to negligible VMR.
            return True
        if len(self.max_change_history) == 0:
            return False
        if self.max_change_history[-1] < self.config.convergence_threshold:
            return True
        recent = self.max_change_history[-4:]
        tight_changes = max(recent) - min(recent) < 0.1 * np.mean(recent)
        close_mean_changes = np.mean(recent) < 2.0 * self.config.convergence_threshold
        return tight_changes and close_mean_changes

    def get_max_change(self) -> float:
        """
        Get current max change for this layer. If the layer is marked as skipped due to negligible VMR, defaults to 0.
        If no change history has been established yet (no iterations have occurred) this defaults to np.inf.
        """
        if self.skipped:
            return 0.0
        if len(self.max_change_history) == 0:
            return np.inf
        return self.max_change_history[-1]


class MultiSpeciesAccelerator:
    """
    Species-aware convergence accelerator for coupled NLTE population systems.

    Manages a :class:`tiramisu.accelerator.LayerAccelerator` per (species, layer) pair for damping/convergence
    bookkeeping, and owns cross-species Anderson extrapolation directly: species on the same atmospheric layer are
    radiatively coupled, so a single extrapolation step is fit jointly across every active species' populations on a
    layer (concatenated in ``self.species_order``) rather than per species independently - the same reasoning that
    already applies to the shared per-layer damping factor (``self._layer_omega``).

    :func:`apply` is the single entry point: given every active species' newly solved (raw) populations for one
    layer, it registers the per-species and per-layer change history, adapts/reuses the shared omega, restarts the
    extrapolation history if the last extrapolated step made the requested change grow, damps each species, attempts
    one merged extrapolation step if eligible, and returns the accepted populations per species.

    Parameters
    ----------
    species_layers : dict[str, int]
        Mapping from species name to the number of NLTE layers for that species. All species must have the same layer
        count in the current implementation (since they share the same atmospheric layer grid), but the interface
        accepts per-species counts for forward compatibility.
    config : AccelerationConfig or None
        Shared acceleration configuration. If None, defaults are used.
    """

    __slots__ = [
        "config", "accelerators", "species_order", "_layer_omega", "_layer_omega_iteration",
        "_layer_raw_history", "_layer_input_history", "_layer_max_change_history",
        "_layer_extrapolation_disabled_until_iter", "_layer_extrapolated_iter",
    ]

    def __init__(
            self,
            species_layers: dict[str, int],
            config: AccelerationConfig = None,
    ):
        if config is None:
            config = AccelerationConfig()
        if config.method not in ("anderson", "damping_only"):
            raise ValueError(f"Unknown acceleration method {config.method!r}: expected 'anderson' or 'damping_only'.")

        self.config = config

        # Fixed order used whenever species' population vectors are concatenated into one per-layer vector (history
        # buffers below, and the merged extrapolation in _try_extrapolation). Sorted by name for a deterministic,
        # reproducible order independent of dict/set iteration order upstream.
        self.species_order: list[str] = sorted(species_layers, key=str)

        # One LayerAccelerator per (species, layer).
        self.accelerators: dict[str, list[LayerAccelerator]] = {
            species: [
                LayerAccelerator(layer_idx=layer_idx, config=config)
                for layer_idx in range(n_layers)
            ]
            for species, n_layers in species_layers.items()
        }

        # Single damping factor per atmospheric layer, shared across all species on that layer.
        # Key: layer_idx. Value: current omega for that layer.
        self._layer_omega: dict[int, float] = {}

        # Tracks the iteration at which each layer's omega was last adapted, so that with multiple species calling
        # apply() for the same layer in the same iteration, the adaptation rule fires exactly once per layer per
        # iteration rather than once per species.
        self._layer_omega_iteration: dict[int, int] = {}

        # Cross-species per-layer state, keyed by layer_idx, one entry per iteration:
        # - raw (pre-damping/pre-extrapolation) solver output g_j and its input x_j (the previous accepted
        #   populations), each concatenated over species: the pairs _apply_extrapolation works from.
        self._layer_raw_history: dict[int, list[npt.NDArray[np.float64]]] = {}
        self._layer_input_history: dict[int, list[npt.NDArray[np.float64]]] = {}
        # - max relative raw change |g_j - x_j| / x_j over active species (the fixed-point residual), which drives
        #   the omega rule and extrapolation eligibility.
        self._layer_max_change_history: dict[int, list[float]] = {}
        # - post-rejection/failure cooldown, mirroring the old per-species extrapolation_disabled_until_iter.
        self._layer_extrapolation_disabled_until_iter: dict[int, int] = {}
        # - iteration of the last accepted extrapolation, so apply() can restart the history if it made the next
        #   requested change grow.
        self._layer_extrapolated_iter: dict[int, int] = {}

    def mark_skipped(self, species: str, layer_idx: int) -> None:
        """Mark a (species, layer) as permanently inactive due to zero/negligible VMR.
        The layer will report max_change=0.0 and converged=True without requiring any apply() calls, and will be
        excluded from that layer's merged extrapolation vector."""
        self.accelerators[species][layer_idx].skipped = True

    def apply(
            self,
            layer_idx: int,
            pop_new: dict[str, npt.NDArray[np.float64]],
            pop_old: dict[str, npt.NDArray[np.float64]],
            iteration: int,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Register and accelerate every active species' newly solved populations for one atmospheric layer.

        ``pop_new``/``pop_old`` must contain exactly the non-skipped species for this layer (one call replaces the
        old register_raw()-then-apply() pair of per-species loops).

        Parameters
        ----------
        layer_idx : int
            NLTE-relative layer index.
        pop_new : dict[str, ndarray]
            Newly solved, normalised populations for every active species on this layer.
        pop_old : dict[str, ndarray]
            Previous-iteration, normalised populations for the same species (from ``NLTEProcessor.solve_pops`` -
            this is *not* sourced from each species' own ``accepted_history``, since at iteration 1 there is no
            accepted history yet, and the aggregated-state indexing a species uses can in principle shift between
            iterations, both of which only ``solve_pops`` can resolve correctly).
        iteration : int
            Current iteration number (0-indexed).

        Returns
        -------
        dict[str, ndarray]
            Accepted (accelerated/damped) populations per species, each normalised to 1.
        """
        species_list = [sp for sp in self.species_order if sp in pop_new]
        layers = [self.accelerators[sp][layer_idx] for sp in species_list]
        floor = self.config.min_relative_pop

        # Per-layer raw history: the fixed-point residual (max over species) drives omega, and the (input, raw
        # output) pairs drive the extrapolation. Every entry is raw, so the omega rule never mixes in accepted changes.
        max_hist = self.config.extrapolation_history_size + 1
        for history, value in (
                (self._layer_max_change_history,
                 max(_compute_max_change(pop_new[sp], pop_old[sp], floor) for sp in species_list)),
                (self._layer_raw_history, np.concatenate([pop_new[sp] for sp in species_list])),
                (self._layer_input_history, np.concatenate([pop_old[sp] for sp in species_list])),
        ):
            hist = history.setdefault(layer_idx, [])
            hist.append(value)
            if len(hist) > max_hist:
                hist.pop(0)

        omega = self._update_layer_omega(layer_idx, iteration)

        # Requested changes since the extrapolation history (re)started, most recent last.
        changes = self._layer_max_change_history[layer_idx][-len(self._layer_raw_history[layer_idx]):]
        patience = self.config.extrapolation_patience
        if (self._layer_extrapolated_iter.get(layer_idx) == iteration - 1 and len(changes) > patience
                and min(changes[-patience:]) > min(changes[:-patience])):
            # Extrapolating, yet the last `patience` iterations never beat the best requested change seen before them:
            # acceleration is doing worse than not accelerating. Typically the stored pairs no longer describe this
            # layer's map (its neighbours move it between iterations) and the extrapolation keeps returning close to
            # old iterates while the requested change grows. Restart from the current pair; Anderson resumes once the
            # window has refilled. A single bad iteration is tolerated: Anderson's requested change is not monotone
            # even when it converges (one-iteration jumps of ~8x on converging linear maps).
            del self._layer_raw_history[layer_idx][:-1]
            del self._layer_input_history[layer_idx][:-1]
            if log.isEnabledFor(_LOG_VERBOSE_2):
                log.log(
                    _LOG_VERBOSE_2,
                    f"[nL{layer_idx}] No improvement on requested change {min(changes[:-patience]):.3e} in the last "
                    f"{patience} iterations (now {changes[-1]:.3e}) while extrapolating; restarting "
                    f"{self.config.method} history."
                )

        if iteration <= self.config.warmup_iterations:
            # Warmup: accept the raw iterate as-is for every species.
            accepted = {sp: pop_new[sp].copy() for sp in species_list}
        else:
            accepted = {
                sp: la._apply_damping(pop_new[sp], pop_old[sp], omega) for sp, la in zip(species_list, layers)
            }
            if self._extrapolation_eligible(layer_idx, iteration):
                accepted = self._try_extrapolation(
                    layer_idx=layer_idx,
                    species_list=species_list,
                    pop_damped=accepted,
                    pop_old=pop_old,
                    iteration=iteration,
                )

        for sp, la in zip(species_list, layers):
            # Convergence is judged on the change the solve requests (raw output vs input: how far the populations are
            # from satisfying their own rate equations), not on the accepted step. With damping the accepted change is
            # omega times the requested one, so this only tightens the threshold by 1/omega; an extrapolated step,
            # though, can be near zero while the requested change is large, which must not count as converged.
            la.max_change_history.append(_compute_max_change(pop_new[sp], pop_old[sp], floor))

        return accepted

    def _should_use_anderson(self, layer_idx: int, iteration: int) -> bool:
        """
        Eligibility gate for attempting Anderson acceleration.

        Deliberately has no "already converging smoothly" pre-check: Anderson (with its wide, adaptively-truncated
        window) is the tool for persistent slow oscillations that damping cannot remove, so such a check would rule
        out exactly the regime it exists to handle.

        The only requirements are (a) enough stored history to fill the candidate window, and (b) not currently in
        a post-failure cooldown. Whether a given extrapolated step is actually trustworthy is decided *after* it is
        computed, by comparing it against the damped baseline in :func:`_accept_extrapolation`, and afterwards by its
        outcome: if accepted steps stop improving the requested change (``config.extrapolation_patience``),
        :meth:`apply` clears the history, so (a) holds again only after ``anderson_window + 1`` fresh iterations.
        """
        if iteration <= self._layer_extrapolation_disabled_until_iter.get(layer_idx, -1):
            return False

        if len(self._layer_raw_history.get(layer_idx, [])) < self.config.anderson_window + 1:
            return False

        if len(self._layer_max_change_history.get(layer_idx, [])) < 4:
            return False

        return True

    def _extrapolation_eligible(self, layer_idx: int, iteration: int) -> bool:
        """Whether this layer would currently attempt merged Anderson extrapolation (never with method "damping_only";
        shared across all species - there is no per-species override)."""
        return self.config.method == "anderson" and self._should_use_anderson(layer_idx, iteration)

    def _try_extrapolation(
            self,
            layer_idx: int,
            species_list: list[str],
            pop_damped: dict[str, npt.NDArray[np.float64]],
            pop_old: dict[str, npt.NDArray[np.float64]],
            iteration: int,
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Attempt one merged extrapolation step across every active species on this layer, mix it with the damped
        baseline, and accept/reject it via the shared safety envelope. Falls back to ``pop_damped`` on rejection or
        failure, and starts the post-failure cooldown in either case.
        """
        sizes = [pop_damped[sp].shape[0] for sp in species_list]
        concat_damped = np.concatenate([pop_damped[sp] for sp in species_list])
        concat_old = np.concatenate([pop_old[sp] for sp in species_list])

        try:
            pop_accel_raw = _apply_extrapolation(
                raw_history=self._layer_raw_history[layer_idx],
                input_history=self._layer_input_history[layer_idx],
                window=self.config.anderson_window,
                max_alpha=self.config.anderson_max_alpha,
                floor=self.config.min_relative_pop,
                svd_rcond=self.config.svd_rcond,
                ridge=self.config.extrapolation_ridge,
                layer_idx=layer_idx,
            )
            # Renormalise per species before mixing (each species' segment is its own probability distribution;
            # the concatenation is only a vehicle for fitting shared extrapolation coefficients across species).
            pop_accel_raw = _renormalize_segments(pop_accel_raw, sizes)
            pop_accel = self.config.extrapolation_mix * pop_accel_raw + (
                        1.0 - self.config.extrapolation_mix) * concat_damped
            pop_accel = _renormalize_segments(pop_accel, sizes)

            if _accept_extrapolation(
                    pop_accel=pop_accel, pop_damped=concat_damped, pop_old=concat_old,
                    config=self.config, layer_idx=layer_idx,
            ):
                if log.isEnabledFor(_LOG_VERBOSE_2):
                    log.log(_LOG_VERBOSE_2, f"[nL{layer_idx}] Accepted {self.config.method} acceleration.")
                # Checked against the next iteration's requested change in apply().
                self._layer_extrapolated_iter[layer_idx] = iteration
                accepted = {}
                offset = 0
                for sp, size in zip(species_list, sizes):
                    accepted[sp] = pop_accel[offset:offset + size]
                    offset += size
                return accepted

            if log.isEnabledFor(_LOG_VERBOSE_2):
                log.log(_LOG_VERBOSE_2, f"[nL{layer_idx}] Rejected {self.config.method} acceleration.")
            self._layer_extrapolation_disabled_until_iter[layer_idx] = (
                    iteration + self.config.extrapolation_disable_iterations
            )

        except RuntimeError as e:
            log.warning(f"[nL{layer_idx}] {self.config.method} failed: {e}.")
            self._layer_extrapolation_disabled_until_iter[layer_idx] = (
                    iteration + self.config.extrapolation_disable_iterations
            )

        return pop_damped

    def _update_layer_omega(self, layer_idx: int, iteration: int) -> float:
        """
        Adapt or reuse the per-layer single damping factor shared by every species.

        Adaptation happens at most once per layer per iteration (``apply()`` is only ever called once per layer per
        iteration itself, but this guard keeps the method idempotent/safe to call more than once).

        The adaptation uses the shared physical rule :func:`_adapt_omega`, driven by cross-species signals:

        - ``oscillating`` is True if the last two requested steps on this layer, r_k = log g_k - log x_k over every
          active species' levels (g_k the raw solve output, x_k its input), point in opposite directions
          (r_k . r_{k-1} < 0): the solve keeps asking to undo its previous step, an overshoot that damping removes.
          Steps that keep their direction are a slow drift, which damping would only slow down (see
          :func:`_adapt_omega`), so a non-monotone change size alone is not treated as oscillation. Raw steps are used,
          not accepted ones: they are what the solve asks for whatever damping or extrapolation was applied.
        - ``extrapolation_active`` is True if this layer is currently eligible to attempt the merged Anderson
          extrapolation step (see :func:`_extrapolation_eligible`), since extrapolation needs a sufficiently
          compressed damped history to stay well-conditioned.

        Parameters
        ----------
        layer_idx : int
            NLTE-relative layer index.
        iteration : int
            Current iteration number (0-indexed).

        Returns
        -------
        float
            The (possibly just-updated) omega for this layer.
        """
        if self._layer_omega_iteration.get(layer_idx) == iteration:
            return self._layer_omega[layer_idx]

        # Confusingly, setdefault gets the value at the layeridx key if it's there, else creates the entry with the
        # default value passed and then returns it.
        old_omega = self._layer_omega.setdefault(layer_idx, self.config.omega_start)
        if iteration <= self.config.warmup_iterations:
            return old_omega

        # Requested steps r_j = log g_j - log x_j of the last two iterations (raw solve output g_j, input x_j).
        floor = self.config.min_relative_pop
        steps = [
            np.log(np.maximum(g, floor)) - np.log(np.maximum(x, floor))
            for g, x in zip(self._layer_raw_history[layer_idx][-2:], self._layer_input_history[layer_idx][-2:])
        ]
        oscillating = (
                len(steps) == 2 and steps[0].shape == steps[1].shape and steps[1] @ steps[0] < 0
                and self._layer_max_change_history[layer_idx][-1] > self.config.convergence_threshold
        )

        extrapolation_active = self._extrapolation_eligible(layer_idx, iteration)

        new_omega = _adapt_omega(
            omega=old_omega, oscillating=oscillating, config=self.config, extrapolation_active=extrapolation_active
        )
        self._layer_omega[layer_idx] = new_omega
        self._layer_omega_iteration[layer_idx] = iteration

        if new_omega != old_omega:
            if log.isEnabledFor(_LOG_VERBOSE_2):
                log.log(
                    _LOG_VERBOSE_2,
                    f"[nL{layer_idx}] Layer omega: {old_omega:.2f}->{new_omega:.2f}"
                    f" ({'oscillating' if oscillating else 'smooth'})"
                )

        return new_omega

    def converged(self, species: str = None) -> bool:
        """Whether every layer of ``species`` has converged, or of every species if it is None."""
        if species is not None:
            return all(la.converged() for la in self.accelerators[species])
        else:
            return all(
                la.converged()
                for las in self.accelerators.values()
                for la in las
            )

    def get_max_changes(self, species: str) -> npt.NDArray[np.float64]:
        """Return per-layer maximum changes for a given species."""
        return np.array([la.get_max_change() for la in self.accelerators[species]])
