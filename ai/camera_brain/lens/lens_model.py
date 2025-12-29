class LensModel:
    """
    Simulates perceptual lens behavior.
    """

    def compute_compression(
        self,
        distance: float,
        compression_bias: float,
    ) -> float:
        """
        Returns parallax compression factor.
        """

        return max(
            0.3,
            min(1.0, 1.0 / (distance * (1.0 - compression_bias)))
        )