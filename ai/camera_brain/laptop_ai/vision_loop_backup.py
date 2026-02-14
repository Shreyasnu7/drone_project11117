    async def _vision_loop(self):
        """
        P1.0: Real-time Vision Loop (Camera -> YOLO -> Render -> Record).
        Runs continuously, independent of server commands.
        """
        print("🔥 Vision Loop Configured (GPU: Active)")
        
        # --- CAMERA & VIDEO SETUP (5.3K / MAX RES) ---
        # 1. Determine optimal resolution (GoPro Hero 12/13/11 or 4K Webcam)
        # Default to 1080p if not specified, but scan for high-res
        CAM_WIDTH, CAM_HEIGHT = 1920, 1080 
        
        cam_stream = None
        try:
            from laptop_ai.threaded_camera import CameraStream
            print("📷 REQUESTING RESOLUTION: 1920x1080 (starting point)")
            self.cam_stream = CameraStream(src=0, width=CAM_WIDTH, height=CAM_HEIGHT).start() 
            actual_width, actual_height = self.cam_stream.width, self.cam_stream.height
            CAM_WIDTH, CAM_HEIGHT = actual_width, actual_height
            print(f"✅ CAMERA NEGOTIATED: {CAM_WIDTH}x{CAM_HEIGHT}")
        except Exception as e:
            print(f"❌ Camera Init Failed: {e}")
            self.cam_stream = None
            CAM_WIDTH, CAM_HEIGHT = 1280, 720 # Fallback for UI
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_path = f"drone_footage_{int(time.time())}.mp4"
        
        # We initialize video_out later or here? 
        # Original code initialized it here.
        video_out = cv2.VideoWriter(out_path, fourcc, 30.0, (CAM_WIDTH, CAM_HEIGHT))
        
        print(f"⏺️  RECORDING TO: {out_path}")
        
        while True:
            if self.cam_stream:
                raw_frame = self.cam_stream.read()
            else:
                raw_frame = None

            if raw_frame is not None:
                # 3. YOLOV8 INFERENCE (Person/Vehicle Detection)
                # Run in separate thread to keep render FPS high
                if self.threaded_yolo:
                    self.threaded_yolo.update(raw_frame)
                    detections = self.threaded_yolo.get_latest_detections()
                else:
                    detections = []

                # 4. GIMBAL & FLIGHT LOGIC (The "Brain")
                if self.tracker:
                     # Update tracker with new detections
                     self.tracker.update(detections, raw_frame)
                     # Get target position
                     target = self.tracker.get_target()
                     
                     if target and self.gimbal_brain:
                         # PID Control for Gimbal
                         pitch, yaw = self.gimbal_brain.compute(target, (CAM_WIDTH, CAM_HEIGHT))
                         # Send to ESP32 via Autopilot/Mavlink
                         # self.autopilot.set_gimbal(pitch, yaw)

                # 5. CINEMATIC COLOR GRADING (Real-time LUTs)
                # Apply Global Tone Curve + LUT
                processed_frame = raw_frame # Placeholder for full pipeline
                
                # 6. UI OVERLAY (HUD)
                # Draw boxes, status, recording indicator
                display_frame = processed_frame.copy()
                
                # Draw Detections
                for box in detections:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    
                # Draw Status
                cv2.putText(display_frame, f"MODE: {self.current_action}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Recording Indicator
                if self.is_recording:
                     cv2.circle(display_frame, (cam_width-50, 50), 20, (0, 0, 255), -1)
                
                # Show
                cv2.imshow("DRONE EYE (Director Core)", display_frame)
                
                # Record
                if self.is_recording:
                    video_out.write(processed_frame)

            else:
                # Show "Connecting" screen
                blank = np.zeros((720, 1280, 3), np.uint8)
                cv2.putText(blank, "SEARCHING FOR CAMERA...", (400, 360),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("DRONE EYE (Director Core)", blank)
                time.sleep(0.1)

            # Key Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        video_out.release()
        cv2.destroyAllWindows()
