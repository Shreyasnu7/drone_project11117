class MotionAnticipator:
    """
    Predicts near-future subject motion to lead camera movement.
    """

    def predict(
        self,
        velocity: float,
        acceleration: float,
        anticipation_bias: float,
    ) -> float:
        """
        Returns predicted motion scalar.
        """

        predicted = velocity + 0.5 * acceleration
        return predicted * (1.0 + anticipation_bias)