
import cv2
import threading
import queue
import time
import os

class AsyncVideoWriter:
    def __init__(self, filename="cinema_output.mp4", fps=60.0):
        self.filename = filename
        self.fps = fps
        self.queue = queue.Queue()
        self.writer = None
        self.running = False
        self.thread = None
        self.width = 0
        self.height = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()
        print(f"🎥 Cinema Recorder Started: {self.filename} @ {self.fps} FPS")

    def write(self, frame):
        if not self.running:
            return
        # Don't let the queue get too huge if disk is slow, better to drop frames than crash RAM
        if self.queue.qsize() < 100: 
            self.queue.put(frame)

    def _write_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:
                continue

            h, w = frame.shape[:2]
            
            # Lazy Init / Re-Init if res changes
            if self.writer is None or w != self.width or h != self.height:
                if self.writer:
                    self.writer.release()
                
                self.width = w
                self.height = h
                
                # mp4v is reliable, avc1 (H.264) is better but requires openh264 on windows sometimes
                # trying mp4v first for compatibility
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (w, h))
                print(f"📼 Writer Initialized: {w}x{h}")

            self.writer.write(frame)
            self.queue.task_done()

        if self.writer:
            self.writer.release()
            print("💾 Video File Saved.")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
