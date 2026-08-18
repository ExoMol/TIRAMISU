import typing as t
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
    omega_min: float = 0.5
    omega_max: float = 1.0
    omega_increase_factor: float = 1.45  # Was 1.05 # Recovering from oscillation; return preference to new step.
    omega_decrease_factor: float = 0.7  # Oscillating; prefer pops from previous iteration.

    # Extrapolation acceleration - shared machinery for both Ng-style and Anderson-style extrapolation. "Ng" and
    # "Anderson" differ only in the size of the candidate window of residual-difference vectors offered; the number
    # actually used each call is determined adaptively via SVD truncation (`svd_rcond`), not fixed in advance - see
    # _solve_extrapolation_coefficients.
    method: str = "anderson"  # Extrapolation method: "ng", "anderson", or "damping_only". Shared across every
    # species/layer - there is no per-species override.
    extrapolation_history_size: int = 15  # Rolling buffer of past iterates/residuals retained. Must be >=
    # max(ng_window, anderson_window) + 1.
    ng_window: int = 2  # Candidate window (# of deltas) for classic Ng-style extrapolation.
    anderson_window: int = 8  # Candidate window (# of deltas) for Anderson-style extrapolation. This is an upper
    # bound on the candidate pool, not a fixed count - see _solve_extrapolation_coefficients for how the effective
    # window is chosen adaptively each call.
    svd_rcond: float = 1e-8  # Relative singular-value cutoff (vs. the largest) used to adaptively truncate the
    # extrapolation subspace to its well-conditioned part.
    extrapolation_ridge: float = 1e-8  # Small additional Tikhonov ridge, scaled to the *largest retained* singular
    # value squared (not the trace/mean - see _solve_extrapolation_coefficients docstring for why that matters).

    # Ng acceleration
    extrapolation_enable_threshold: float = 5e-3  # 0.5% - turn on when converging smoothly.
    extrapolation_disable_iterations: int = 5  # Disable for N iters after failure.
    ng_max_alpha: float = 3.0  # Coefficient-magnitude guard for Ng's small (2-vector) window.
    anderson_max_alpha: float = 50.0  # Separate, more permissive guard for Anderson's larger window: legitimately
    # well-conditioned solutions naturally need larger, sign-alternating coefficients to cancel a many-candidate,
    # partially-redundant history - this is not itself a sign of instability the way it would be for Ng's 2-vector
    # case. The physically-grounded step-size/change checks in _accept_extrapolation remain the real safety net.
    extrapolation_step_limit: float = 2.0
    extrapolation_worsening_factor: float = 2.0  # 1.2
    min_relative_pop: float = 1e-70
    extrapolation_mix: float = 1.0  # 0.5
    extrapolation_omega_cap = 0.8

    # Convergence
    convergence_threshold: float = 1e-3  # 0.1%.

    # Safety
    check_max_jump: bool = True
    max_jump_factor: float = 5.0
    min_pop_for_jump_check: float = 1e-6


def _adapt_omega(
        omega: float,
        oscillating: bool,
        config: "AccelerationConfig",
        extrapolation_active: bool,
) -> float:
    """
    Pure update rule for the adaptive damping factor omega.

    Encodes the damping rules: if the relevant change history is oscillating (non-monotonic), omega is decreased
    (bounded below by ``config.omega_min``) to favour the previous iterate more strongly. Otherwise, omega is relaxed
    back towards 1 (bounded above by ``config.omega_max``, or by the tighter ``config.ng_omega_cap`` when Ng/DIIS
    acceleration is active, since Ng needs a sufficiently compressed damped history to remain well-conditioned).

    This function is stateless: it takes the current omega and returns the next value. Used by
    :class:`MultiSpeciesAccelerator` for the single omega shared across all species on a given atmospheric layer.

    Parameters
    ----------
    omega : float
        Current value of the damping factor.
    oscillating : bool
        Whether the aggregated per-layer change history is non-monotonic.
    config : AccelerationConfig
        Shared acceleration configuration.
    extrapolation_active : bool
        Whether Ng/DIIS acceleration is eligible/active for the relevant accelerator(s) this iteration. Used only
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
        deltas: npt.NDArray[np.float64],
        max_alpha: float,
        svd_rcond: float,
        extrapolation_ridge: float,
        layer_idx: int,
) -> npt.NDArray[np.float64]:
    """
    Solve the DIIS/Anderson-type constrained least-squares problem

    .. math::
        \\alpha = \\arg\\min_a \\|D a\\|^2 \\quad \\text{subject to} \\quad \\sum_i a_i = 1,

    where the columns of :math:`D` are the supplied residual-difference vectors ``deltas`` (one row per delta).

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
    generous candidate pool (``deltas`` can have as many rows as ``config.anderson_window``/``config.ng_window``)
    and retains only the singular values that are numerically significant relative to the largest one
    (``singular_value > config.svd_rcond * largest_singular_value``). Any candidate vectors beyond the problem's
    true rank - which is bounded by the number of independent population directions, i.e. at most
    ``n_states - 1`` given the sum-to-one constraint, and will typically be far smaller than a large candidate
    window like ``anderson_window`` - are automatically discarded rather than forced into an ill-conditioned
    solve. A small additional ridge, scaled to the *largest retained* singular value squared, is applied on top
    of the truncation for extra smoothing.

    Parameters
    ----------
    deltas : ndarray, shape (m, n_states)
        Candidate residual-difference (delta) vectors, one per row, most recent last.
    max_alpha : float
        Coefficient-magnitude guard (``config.ng_max_alpha`` for Ng's small window, ``config.anderson_max_alpha``
        for Anderson's larger window - a larger window structurally needs larger cancelling coefficients even
        when well-conditioned, so the two are not comparable and shouldn't share one threshold).
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
    # D: n_states x m, columns are the delta vectors.
    d_matrix = deltas.T
    num_candidates = d_matrix.shape[1]

    # Economy SVD - k = min(n_states, m) singular values/vectors.
    u, s, vt = np.linalg.svd(d_matrix, full_matrices=False)

    if s[0] <= 0 or not np.isfinite(s[0]):
        raise RuntimeError("Degenerate delta matrix (zero or non-finite leading singular value).")

    mask = s > svd_rcond * s[0]

    if not np.any(mask):
        raise RuntimeError("No singular values survived truncation - delta matrix is numerically null.")

    s_trunc = s[mask]
    v_trunc = vt[mask, :].T  # m x k'

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
        window: int,
        max_alpha: float,
        floor: float,
        svd_rcond: float,
        ridge: float,
        pop_new: npt.NDArray[np.float64],
        layer_idx: int,
) -> npt.NDArray[np.float64]:
    """
    Shared core for both Ng-style (small, fixed candidate window) and Anderson-style (larger, adaptively-truncated
    candidate window) extrapolation in logarithmic population space, using the current nonlinear iteration together
    with a history of previously accepted iterates.

    The method constructs an accelerated population vector from the most recent ``window + 1`` iterates. The current
    nonlinear solution (``pop_new``) is temporarily appended to the stored history before constructing the
    acceleration system.

    The acceleration is performed in logarithmic population space:

    .. math::
        y_k = \\log(\\max(n_k, \\epsilon)),

    where :math:`n_k` is the population vector for iteration :math:`k`, and :math:`\\epsilon` is a small floor
    value to avoid singularities for very small populations. Differences are then constructed as
    :math:`d_k = y_{k+1} - y_k`, and the extrapolation coefficients :math:`\\alpha` are obtained from
    :func:`_solve_extrapolation_coefficients`. The accelerated logarithmic population vector is:

    .. math::
        y_{\\mathrm{ext}} = y_m + \\sum_i \\alpha_i d_i,

    where :math:`y_m` is the logarithm of the current nonlinear iterate. Finally, the accelerated populations
    are recovered via exponentiation and normalization. Performing the extrapolation in logarithmic space
    improves robustness for NLTE population problems where state populations may span many orders of magnitude.

    Build the working history from raw_history - the sequence of raw (pre-damping/pre-extrapolation) solver
    iterates, populated once per iteration in ``MultiSpeciesAccelerator.apply()``'s phase-1 bookkeeping. This is
    deliberately NOT the accepted-population history (which stores the *accepted*, i.e. already-damped/extrapolated,
    populations): mixing a heavily-damped history with a raw
    current iterate produces deltas whose scale differs by roughly 1/omega between the historical steps and the
    newest one whenever omega is small - exactly the regime this method exists to help with - and the resulting
    extrapolated step inherits that inflated scale, guaranteeing rejection by the ng_step_limit safety check.
    Working entirely from the raw, undamped trajectory keeps every delta on the same footing, which is also what
    Anderson/DIIS acceleration assumes: it operates on the underlying nonlinear fixed-point map's own iterates, with
    damping/mixing applied only as a separate acceptance safeguard afterwards (see _try_extrapolation), not folded
    into the history being extrapolated.

    Parameters
    ----------
    raw_history : ndarray,
    window : int,
    max_alpha : float,
    floor : float,
    svd_rcond : float,
    ridge : float,
    pop_new : np.ndarray
        Newly computed population vector for the current nonlinear iteration prior to damping or acceleration
        acceptance.
    layer_idx : int

    Returns
    -------
    pop_ext : np.ndarray
        Extrapolated population vector in raw (unnormalised) population space. Unlike the previous single-species
        contract, this is deliberately left unnormalised: callers building a multi-species vector (see
        :class:`MultiSpeciesAccelerator`) need to renormalise each species' segment independently, since the
        concatenation is not itself a single probability distribution - see :func:`_renormalize_segments`.

    Raises
    ------
    RuntimeError
        Propagated from :func:`_solve_extrapolation_coefficients`, or raised if the result is non-finite.
    """
    num_hist = window + 1

    # History is the raw history, pre-damping/extrapolation. The phase-1 bookkeeping in
    # MultiSpeciesAccelerator.apply() always appends the current pop_new before this is called, so raw_history
    # already ends with it - no need to append it again here.
    if not raw_history or not np.array_equal(raw_history[-1], pop_new):
        raise RuntimeError(
            "raw_history is out of sync with pop_new - the current iterate must already be the last entry in "
            "raw_history before apply() is called for this iteration."
        )

    pops = np.array(raw_history[-num_hist:])

    # Transform to log-space.
    log_pops = np.log(np.maximum(pops, floor))
    # Residuals in log-space
    deltas = np.diff(log_pops, axis=0)

    alpha = _solve_extrapolation_coefficients(
        deltas=deltas, max_alpha=max_alpha, svd_rcond=svd_rcond, extrapolation_ridge=ridge, layer_idx=layer_idx
    )

    # Extrapolate populations from alpha coefficients and differences.
    log_ext = log_pops[-1].copy()  # This is the log of pop_new.

    for a, log_dif in zip(alpha, deltas):
        log_ext += a * log_dif

    # Enforce positivity. Normalisation is the caller's responsibility (see docstring).
    pop_ext = np.maximum(np.exp(log_ext), 0.0)
    if not np.isfinite(pop_ext).all():
        raise RuntimeError("Extrapolation produced non-finite populations.")

    return pop_ext


def _renormalize_segments(vec: npt.NDArray[np.float64], sizes: t.List[int]) -> npt.NDArray[np.float64]:
    """
    Renormalise each contiguous per-species segment of a concatenated population vector to sum to 1.

    ``vec`` is a concatenation (in a fixed species order) of several independent population distributions - one per
    species - stacked together only so that a single Ng/Anderson extrapolation step can be fit jointly across all of
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
        min_relative_pop: float,
        layer_idx: int,
) -> bool:
    """Shared accept/reject safety envelope for an extrapolated step, vs. its damped baseline."""
    if np.any(~np.isfinite(pop_accel)):
        log.log(_LOG_VERBOSE_2, "Rejected extrapolation due to infinite populations.")
        return False

    if np.any(pop_accel < 0):
        log.log(_LOG_VERBOSE_2, "Rejected extrapolation due to negative populations.")
        return False

    accel_step = np.linalg.norm(pop_accel - pop_old)
    damped_step = np.linalg.norm(pop_damped - pop_old)

    if accel_step > config.extrapolation_step_limit * damped_step:
        log.log(
            _LOG_VERBOSE_2,
            f"[nL{layer_idx}] Rejected extrapolation as step 2-norm more than {config.extrapolation_step_limit}"
            f" times damped."
        )
        return False

    accel_change = _compute_max_change(pop_accel, pop_old, min_relative_pop)
    damped_change = _compute_max_change(pop_damped, pop_old, min_relative_pop)

    if accel_change > config.extrapolation_worsening_factor * damped_change:
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
    it. Ng/Anderson extrapolation is no longer performed here: species on the same layer are radiatively coupled, so
    it is fit jointly across all of them by :class:`MultiSpeciesAccelerator` instead (see its ``apply``) - a single
    species' isolated history is not enough context to extrapolate against.
    """

    __slots__ = ["layer_idx", "config", "max_change_history", "accepted_history", "skipped"]

    def __init__(self, layer_idx: int, config: AccelerationConfig):
        self.layer_idx = layer_idx
        self.config = config
        # Skipped is False, unless marked as True during runtime when the species' VMR is below a VMR threshold.
        self.skipped = False

        # History storage - grows dynamically as iterations proceed.
        self.max_change_history: t.List[float] = []  # One float per iteration.
        self.accepted_history: t.List[npt.NDArray[np.float64]] = []  # One array per iteration.

    def _store_iteration(self, pop: npt.NDArray[np.float64]) -> None:
        self.accepted_history.append(pop.copy())

        max_hist = self.config.extrapolation_history_size + 1
        if len(self.accepted_history) > max_hist:
            self.accepted_history.pop(0)

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
        # return self.change_history[-1] < self.config.convergence_threshold
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
    bookkeeping, and owns cross-species Ng/Anderson extrapolation directly: species on the same atmospheric layer are
    radiatively coupled, so a single extrapolation step is fit jointly across every active species' populations on a
    layer (concatenated in ``self.species_order``) rather than per species independently - the same reasoning that
    already applies to the shared per-layer damping factor (``self._layer_omega``).

    :func:`apply` is the single entry point: given every active species' newly solved (raw) populations for one
    layer, it registers the per-species and per-layer change history, adapts/reuses the shared omega, damps each
    species, attempts one merged extrapolation step if eligible, and returns the accepted populations per species.

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
        "species_layers", "config", "accelerators", "species_order", "_layer_omega", "_layer_omega_iteration",
        "_layer_raw_history", "_layer_max_change_history", "_layer_extrapolation_disabled_until_iter",
    ]

    def __init__(
            self,
            species_layers: dict[str, int],
            config: AccelerationConfig = None,
    ):
        if config is None:
            config = AccelerationConfig()

        self.config = config
        self.species_layers = species_layers

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

        # Cross-species state for the merged Ng/Anderson extrapolation, keyed by layer_idx:
        # - raw (pre-damping/pre-extrapolation) concatenated population vectors, one per iteration - the buffer
        #   _apply_extrapolation builds deltas from (see its docstring for why raw, not accepted, history is used).
        self._layer_raw_history: dict[int, list[npt.NDArray[np.float64]]] = {}
        # - aggregated (max-over-active-species) max relative change, one per iteration - drives extrapolation
        #   eligibility the same way a single species' own max_change_history used to.
        self._layer_max_change_history: dict[int, list[float]] = {}
        # - post-rejection/failure cooldown, mirroring the old per-species extrapolation_disabled_until_iter.
        self._layer_extrapolation_disabled_until_iter: dict[int, int] = {}

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

        # Phase 1: per-species and aggregated per-layer change/history bookkeeping.
        layer_max_change = 0.0
        for sp, la in zip(species_list, layers):
            max_change = _compute_max_change(pop_new[sp], pop_old[sp], floor)
            la.max_change_history.append(max_change)
            layer_max_change = max(layer_max_change, max_change)

        max_hist = self.config.extrapolation_history_size + 1

        layer_changes = self._layer_max_change_history.setdefault(layer_idx, [])
        layer_changes.append(layer_max_change)
        if len(layer_changes) > max_hist:
            layer_changes.pop(0)

        raw_hist = self._layer_raw_history.setdefault(layer_idx, [])
        raw_hist.append(np.concatenate([pop_new[sp] for sp in species_list]))
        if len(raw_hist) > max_hist:
            raw_hist.pop(0)

        omega = self._update_layer_omega(layer_idx, iteration)

        # Warmup: accept the raw iterate as-is for every species.
        if iteration <= self.config.warmup_iterations:
            accepted = {sp: pop_new[sp].copy() for sp in species_list}
            for sp, la in zip(species_list, layers):
                la._store_iteration(accepted[sp])
            return accepted

        pop_damped = {sp: la._apply_damping(pop_new[sp], pop_old[sp], omega) for sp, la in zip(species_list, layers)}
        accepted = pop_damped

        if self._extrapolation_eligible(layer_idx, iteration):
            accepted = self._try_extrapolation(
                layer_idx=layer_idx,
                species_list=species_list,
                pop_damped=pop_damped,
                pop_old=pop_old,
                iteration=iteration,
            )

        for sp, la in zip(species_list, layers):
            la._store_iteration(accepted[sp])
            # Crucial: the convergence check must see the *accepted* (post-damping/extrapolation) change, not the
            # raw one already stored in phase 1 - otherwise it could report convergence on changes that were never
            # actually accepted.
            la.max_change_history[-1] = _compute_max_change(accepted[sp], pop_old[sp], floor)

        return accepted

    def _should_use_ng(self, layer_idx: int, iteration: int) -> bool:
        if iteration <= self._layer_extrapolation_disabled_until_iter.get(layer_idx, -1):
            return False

        raw_hist = self._layer_raw_history.get(layer_idx, [])
        if len(raw_hist) < self.config.extrapolation_history_size:
            return False

        changes = self._layer_max_change_history.get(layer_idx, [])
        if len(changes) < 4:
            return False

        if changes[-1] > self.config.extrapolation_enable_threshold:
            return False

        # Strict monotonic convergence over the most recent 4 iterations, with roughly linear (geometric) decay -
        # see the single-species version this was ported from for the reasoning; behaviour is unchanged, just
        # evaluated on the aggregated per-layer change instead of one species' own.
        recent = changes[-4:]
        monotonic = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))
        if not monotonic:
            return False

        ratios = [recent[i + 1] / recent[i] for i in range(len(recent) - 1)]
        if max(ratios) - min(ratios) > 0.3:
            return False

        return True

    def _should_use_anderson(self, layer_idx: int, iteration: int) -> bool:
        """
        Eligibility gate for attempting Anderson acceleration.

        Deliberately looser than :func:`_should_use_ng`: Ng's monotonic, near-linear pre-check exists because
        classic Ng targets smooth geometric convergence and is fragile outside it, but Anderson (with its wider,
        adaptively-truncated window) is specifically the tool being reached for *because* damping and Ng cannot
        touch a persistent slow oscillation - gating it on the same "already converging smoothly" pre-check would
        rule out exactly the regime it exists to handle.

        The only requirements are (a) enough stored history to fill the candidate window, and (b) not currently in
        a post-failure cooldown. Whether a given extrapolated step is actually trustworthy is decided *after* it is
        computed, by comparing it against the damped baseline in :func:`_accept_extrapolation`.
        """
        if iteration <= self._layer_extrapolation_disabled_until_iter.get(layer_idx, -1):
            return False

        if len(self._layer_raw_history.get(layer_idx, [])) < self.config.anderson_window + 1:
            return False

        if len(self._layer_max_change_history.get(layer_idx, [])) < 4:
            return False

        return True

    def _extrapolation_eligible(self, layer_idx: int, iteration: int) -> bool:
        """Whether this layer would currently attempt merged Ng or Anderson extrapolation, per the configured
        method (shared across all species - there is no per-species override)."""
        if self.config.method == "anderson":
            return self._should_use_anderson(layer_idx, iteration)
        if self.config.method == "ng":
            return self._should_use_ng(layer_idx, iteration)
        return False

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
            if self.config.method == "ng":
                window, max_alpha = self.config.ng_window, self.config.ng_max_alpha
            elif self.config.method == "anderson":
                window, max_alpha = self.config.anderson_window, self.config.anderson_max_alpha
            else:
                raise RuntimeError(f"Unknown extrapolation method {self.config.method!r}; only 'ng' or 'anderson'.")

            pop_accel_raw = _apply_extrapolation(
                raw_history=self._layer_raw_history[layer_idx],
                window=window,
                max_alpha=max_alpha,
                floor=self.config.min_relative_pop,
                svd_rcond=self.config.svd_rcond,
                ridge=self.config.extrapolation_ridge,
                pop_new=self._layer_raw_history[layer_idx][-1],
                layer_idx=layer_idx,
            )
            # Renormalise per species before mixing (each species' segment is its own probability distribution;
            # the concatenation is only a vehicle for fitting shared extrapolation coefficients across species).
            pop_accel_raw = _renormalize_segments(pop_accel_raw, sizes)
            pop_accel = self.config.extrapolation_mix * pop_accel_raw + (1.0 - self.config.extrapolation_mix) * concat_damped
            pop_accel = _renormalize_segments(pop_accel, sizes)

            if _accept_extrapolation(
                    pop_accel=pop_accel, pop_damped=concat_damped, pop_old=concat_old,
                    config=self.config, min_relative_pop=self.config.min_relative_pop, layer_idx=layer_idx,
            ):
                log.log(_LOG_VERBOSE_2, f"[nL{layer_idx}] Accepted {self.config.method} acceleration.")
                accepted = {}
                offset = 0
                for sp, size in zip(species_list, sizes):
                    accepted[sp] = pop_accel[offset:offset + size]
                    offset += size
                return accepted

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

        - ``oscillating`` is True if the *aggregated* (max-over-active-species) recent change on this layer is
          non-monotonic, since damping needs to unify across species rather than let one species' smooth history
          outrun another's oscillating one.
        - ``extrapolation_active`` is True if this layer is currently eligible to attempt the merged Ng/Anderson
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

        # Confusingly, setdeault gets the value at the layeridx key if it's there, else creates the entry with the
        # default value passed and then returns it.
        old_omega = self._layer_omega.setdefault(layer_idx, self.config.omega_start)
        if iteration <= self.config.warmup_iterations:
            return old_omega

        n_recent = 3
        sufficient_history = all(
            len(las[layer_idx].max_change_history) >= n_recent or las[layer_idx].skipped
            for las in self.accelerators.values()
        )
        oscillating = False
        if sufficient_history:
            recent_max_changes = [
                max(
                    0.0 if la[layer_idx].skipped else la[layer_idx].max_change_history[idx]
                    for la in self.accelerators.values()
                )
                for idx in range(-n_recent, 0)
            ]
            oscillating = not all(
                recent_max_changes[i] >= recent_max_changes[i + 1] for i in range(len(recent_max_changes) - 1)
            ) and max(recent_max_changes) > self.config.convergence_threshold

        extrapolation_active = self._extrapolation_eligible(layer_idx, iteration)

        new_omega = _adapt_omega(
            omega=old_omega, oscillating=oscillating, config=self.config, extrapolation_active=extrapolation_active
        )
        self._layer_omega[layer_idx] = new_omega
        self._layer_omega_iteration[layer_idx] = iteration

        if new_omega != old_omega:
            log.log(
                _LOG_VERBOSE_2,
                f"[nL{layer_idx}] Layer omega: {old_omega:.2f}->{new_omega:.2f}"
                f" ({'oscillating' if oscillating else 'smooth'})"
            )

        return new_omega

    def reset_oscillation_flags(self) -> None:
        """
        Kept for the existing per-iteration call site in ``XSecCollection.compute_opacities_profile``; oscillation
        is now derived fresh each call from ``max_change_history``/``_layer_max_change_history`` rather than a
        flag that needed resetting, so there is nothing left to clear.
        """

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
