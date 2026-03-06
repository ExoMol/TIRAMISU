import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)


@dataclass
class AccelerationConfig:
    """Configuration for acceleration methods."""
    # Startup behavior
    warmup_iterations: int = 5  # No acceleration for first N iterations

    # Adaptive damping
    omega_start: float = 1.0
    omega_min: float = 0.5
    omega_max: float = 1.0
    omega_increase_factor: float = 1.1  # Recovering from oscillation; return preference to new step.
    omega_decrease_factor: float = 0.7  # Oscillating; prefer pops from previous iteration.

    # Ng acceleration
    ng_history_length: int = 4
    ng_enable_threshold: float = 0.02  # 2% - turn on when converging smoothly.
    ng_disable_iterations: int = 3  # Disable for N iters after failure

    # Convergence
    convergence_threshold: float = 1e-3  # 0.1%

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
                 "last_iteration", ]

    def __init__(self, layer_idx: int, config: AccelerationConfig):
        self.layer_idx = layer_idx
        self.config = config

        # History storage - grows dynamically as iterations proceed
        # Each element is the max change for that iteration
        self.change_history = []  # List of floats, one per iteration

        # Ng history - stores population arrays
        self.ng_history = []  # List of arrays, one per iteration

        # State
        self.omega = config.omega_start
        self.ng_disabled_until_iter = -1
        self.last_iteration = -1

    def update(
            self,
            pop_new: npt.NDArray[np.float64],
            pop_old: npt.NDArray[np.float64],
            iteration: int,
    ) -> npt.NDArray[np.float64]:
        """
        Apply acceleration/damping for this layer.

        Parameters:
            pop_new: Newly solved populations
            pop_old: Previous iteration populations
            iteration: Current iteration number (0-indexed)

        Returns:
            Accelerated/damped populations
        """

        # Check iteration sequence
        if iteration != self.last_iteration + 1 and iteration != self.last_iteration:
            log.warning(f"[L{self.layer_idx}] Non-sequential iteration (I{self.last_iteration}->{iteration})")

        # Calculate change
        with np.errstate(divide='ignore', invalid='ignore'):
            max_change = np.max(np.abs((pop_new - pop_old) / (pop_old + 1e-30)))

        # Store change history (extend if new iteration)
        if iteration >= len(self.change_history):
            self.change_history.append(max_change)
        else:
            self.change_history[iteration] = max_change

        # Store Ng history
        if iteration >= len(self.ng_history):
            self.ng_history.append(pop_new.copy())
        else:
            self.ng_history[iteration] = pop_new.copy()

        if iteration < self.config.warmup_iterations:
            log.debug(f"[L{self.layer_idx}] Warmup - no acceleration (max. change={max_change:.4e})")
            self.last_iteration = iteration
            return pop_new

        # Try Ng acceleration first (if conditions met)
        if self._should_use_ng(iteration):
            try:
                pop_ng = self._apply_ng()

                if self._ng_is_safe(pop_ng, pop_old):
                    log.info(f"[L{self.layer_idx}] Ng acceleration (max. change={max_change:.4e})")

                    self.last_iteration = iteration
                    return pop_ng
                else:
                    log.warning(f"[L{self.layer_idx}] Ng unsafe, falling back to damping")

                    self.ng_disabled_until_iter = iteration + self.config.ng_disable_iterations

            except RuntimeError as e:
                log.warning(f"[L{self.layer_idx}] Ng failed: {e}, using damping.")
                self.ng_disabled_until_iter = iteration + self.config.ng_disable_iterations

        pop_damped = self._apply_damping(pop_new, pop_old)

        log.debug(f"[L{self.layer_idx}] Damping omega={self.omega:.3f} (max. change={max_change:.4e})")

        self.last_iteration = iteration
        return pop_damped

    def _should_use_ng(self, iteration: int) -> bool:
        """Check if Ng should be attempted."""
        # Disabled temporarily?
        if iteration <= self.ng_disabled_until_iter:
            return False

        # Need enough history
        if len(self.ng_history) < self.config.ng_history_length + 1:
            return False

        # Need to be in smooth convergence regime
        if len(self.change_history) < 3:
            return False

        current_change = self.change_history[-1]
        if current_change > self.config.ng_enable_threshold:
            return False

        # Check for monotonic decrease (smooth convergence)
        recent = self.change_history[-3:]
        monotonic = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))

        return monotonic

    def _apply_ng(self) -> npt.NDArray[np.float64]:
        """Apply Ng acceleration."""
        n = self.config.ng_history_length

        # Extract last n+1 iterates
        pops = np.array(self.ng_history[-(n + 1):])

        # Calculate differences
        deltas = np.diff(pops, axis=0)

        # Build matrix A_ij = delta_n^i · delta_n^j
        a_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a_matrix[i, j] = np.dot(deltas[i], deltas[j])

        # Solve for weights
        ones = np.ones(n)
        reg = 1e-12 * np.trace(a_matrix)
        alpha = np.linalg.solve(a_matrix + reg * np.eye(n), ones)
        alpha /= alpha.sum()

        # Extrapolate
        n_new = np.sum([alpha[i] * pops[i] for i in range(n)], axis=0)

        if not np.isfinite(n_new).all():
            raise RuntimeError("Ng produced non-finite values")

        # Clamp and normalize
        n_new = np.maximum(n_new, 0.0)
        n_new /= n_new.sum()

        return n_new

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
                    log.info(f"[L{self.layer_idx}] Oscillating - omega={old_omega:.2f}->{self.omega:.2f}")
            else:
                # Monotonic - reduce damping (approach omega=1).
                if self.omega < self.config.omega_max:
                    old_omega = self.omega
                    self.omega = min(
                        self.omega * self.config.omega_increase_factor,
                        self.config.omega_max
                    )
                    if self.omega > old_omega:
                        log.debug(f"[L{self.layer_idx}] Smooth - omega={old_omega:.2f}->{self.omega:.2f}")

        pop_damped = self.omega * pop_new + (1 - self.omega) * pop_old

        if np.any(pop_damped < 0):
            log.warning(f"[L{self.layer_idx}] Damping produced negatives, clamping.")
            pop_damped = np.maximum(pop_damped, 0.0)

        pop_damped /= pop_damped.sum()

        return pop_damped

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
