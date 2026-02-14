
import torch
import numpy as np
import cv2

class GPUACESToneCurve:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🚀 ACES Engine Initialized on: {self.device}")
        
        # ACES Input Transform (approximate sRGB -> ACEScg)
        # Using a standard 3x3 matrix for sRGB linear to ACEScg
        self.srgb_to_aces_mat = torch.tensor([
            [0.6131, 0.3395, 0.0474],
            [0.0701, 0.9163, 0.0136],
            [0.0206, 0.1096, 0.8698]
        ], device=self.device, dtype=torch.float32).t() # Transpose formatmul

        # ACES RRT + ODT Approximation (Hill-Langmuir equation style curve)
        # a * (x + b) / (x + c)
        self.a = 2.51
        self.b = 0.03
        self.c = 2.43
        self.d = 0.59
        self.e = 0.14

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Takes a BGR uint8 numpy frame, applies ACES on GPU, returns BGR uint8 numpy frame.
        """
        if frame_bgr is None:
            return None

        # 1. Upload to GPU (H, W, 3) -> (Batch, 3, H, W) is typical for NN, 
        # but for pixel-wise math (H, W, 3) is fine if we use linear algebra carefully.
        # Float conversion happens here.
        # Normalize 0-1
        t_frame = torch.from_numpy(frame_bgr).to(self.device, dtype=torch.float32, non_blocking=True).div_(255.0)

        # 2. Linearize (Swap BGR to RGB first if strictly needed for color accuracy, 
        # ACES is perceptually weird if R and B are swapped).
        # OpenCV is BGR. ACES expects RGB.
        t_frame = t_frame[..., [2, 1, 0]] # BGR -> RGB
        
        # De-gamma (Approximate sRGB -> Linear)
        # Simple pov 2.2 for performance, or exact sRGB piece-wise
        mask = t_frame <= 0.04045
        t_frame[mask] = t_frame[mask] / 12.92
        t_frame[~mask] = torch.pow((t_frame[~mask] + 0.055) / 1.055, 2.4)

        # 3. Apply ACES (RRT+ODT fit)
        # x * (a*x + b) / (x * (c*x + d) + e)
        x = t_frame
        aces_out = (x * (self.a * x + self.b)) / (x * (self.c * x + self.d) + self.e)

        # 4. Clamp & Gamma Correct (Linear -> sRGB Display)
        aces_out = torch.clamp(aces_out, 0.0, 1.0)
        
        # Re-gamma 
        mask_out = aces_out <= 0.0031308
        aces_out[mask_out] = aces_out[mask_out] * 12.92
        aces_out[~mask_out] = 1.055 * torch.pow(aces_out[~mask_out], 1.0 / 2.4) - 0.055

        # 5. RGB -> BGR
        aces_out = aces_out[..., [2, 1, 0]]

        # 6. Download to CPU
        # Scale to 255
        res_uint8 = aces_out.mul_(255.0).clamp_(0, 255).byte().cpu().numpy()
        
        return res_uint8
