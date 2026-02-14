class ExponentialSmoother:
    """
    Simple EMA smoother for drone motion values.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.val = None

    def update(self, measurement):
        if self.val is None:
            self.val = measurement
        else:
            self.val = self.alpha * measurement + (1.0 - self.alpha) * self.val
        return self.val