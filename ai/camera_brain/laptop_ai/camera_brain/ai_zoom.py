class AIZoomComposer:

    def compute_zoom(self, bbox, frame_size):
        x1,y1,x2,y2 = bbox
        width = x2 - x1
        if width < frame_size[0] * 0.2:
            return 1.5
        if width < frame_size[0] * 0.3:
            return 1.2
        return 1.0
