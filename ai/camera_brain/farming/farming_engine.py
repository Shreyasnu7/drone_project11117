class FarmingEngine:
    """
    Analyzes visual 'yield' or composition quality.
    """
    def compute_harvest(self, saliency_map):
        if saliency_map is None: return 0.0
        # Real Logic: Average Saliency Intensity
        # Assuming saliency_map is normalized 0.0-1.0
        import numpy as np
        if hasattr(saliency_map, 'mean'):
            return float(saliency_map.mean())
        return 0.5 # Default neutral if format unknown