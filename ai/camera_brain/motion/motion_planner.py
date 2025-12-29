from ..core.state import CameraBrainState


class MotionPlanner:
    """
    Generates camera-space motion vectors.
    """

    def plan(
        self,
        state: CameraBrainState,
        target_motion: float,
        smoothness: float,
    ) -> tuple[float, float]:
        """
        Returns camera motion vector (x, y).
        """

        dx = (
            smoothness * state.last_motion_vector_x
            + (1.0 - smoothness) * target_motion
        )

        dy = 0.0

        state.last_motion_vector_x = dx
        state.last_motion_vector_y = dy

        return dx, dy