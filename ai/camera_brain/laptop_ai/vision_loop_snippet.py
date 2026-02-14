    async def _vision_loop(self):
        """
        P1.0: Real-time Vision Loop (Camera -> YOLO -> Render -> Record).
        Runs continuously, independent of server commands.
        YIELDS to asyncio loop to allow networking.
        """
        print("🔥 Vision Loop Starting...")
        
        # --- CAMERA SETUP ---
        CAM_WIDTH, CAM_HEIGHT = 1920, 1080 
        
        try:
            from laptop_ai.threaded_camera import CameraStream
            print("📷 REQUESTING RESOLUTION: 1920x1080")
            # Initialize self.cam_stream
            self.cam_stream = CameraStream(src=0, width=CAM_WIDTH, height=CAM_HEIGHT).start() 
            print(f"✅ CAMERA NEGOTIATED: {self.cam_stream.width}x{self.cam_stream.height}")
        except Exception as e:
            print(f"❌ Camera Init Failed: {e} (Will retry in loop)")
            self.cam_stream = None
        
        # Video Writer Setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_path = f"drone_footage_auto.mp4"
        video_out = None # Created on first frame
        
        print(f"🔥 GPU INFERENCE ENGINE: STANDBY (Waiting for frames)")
        
        frame_id = 0
        
        while True:
            t0 = time.time()
            
            # 1. READ FRAME
            raw_frame = None
            if self.cam_stream:
                raw_frame = self.cam_stream.read()
            
            if raw_frame is not None:
                # Resize check (if stream changed)
                if video_out is None:
                     h, w = raw_frame.shape[:2]
                     video_out = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
                     print(f"⏺️  Recording Started: {out_path} ({w}x{h})")

                # 2. AI INFERENCE (YOLO)
                detections = []
                if self.threaded_yolo:
                    self.threaded_yolo.update(raw_frame)
                    detections = self.threaded_yolo.get_latest_detections()
                    
                    if frame_id == 0:
                        print(f"🔥 GPU INFERENCE RUNNING: {len(detections)} objects detected")

                # 3. GIMBAL / TRACKER UPDATE
                if self.tracker and self.gimbal_brain:
                     self.tracker.update(detections, raw_frame)
                     # target = self.tracker.get_target()
                     # if target: ...
                
                # 4. RENDER UI
                display_frame = raw_frame.copy()
                
                # Draw detections
                for box in detections:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    label = f"{self.classifier.names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
                    cv2.putText(display_frame, label, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw Status OSD
                status_text = f"MODE: {self.current_action} | GPU: ON | FPS: {1.0/(time.time()-t0+1e-9):.1f}"
                cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.imshow("Laptop AI Director (GPU Accelerated)", display_frame)
                
                if self.is_recording and video_out:
                    video_out.write(raw_frame)
                
                frame_id += 1

            else:
                # No Camera -> Show Disconnected Screen
                blank = np.zeros((720, 1280, 3), np.uint8)
                cv2.putText(blank, "SEARCHING FOR CAMERA...", (400, 360),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Laptop AI Director (GPU Accelerated)", blank)
                
                # Try to reconnect every 2s?
                # For now just wait
            
            # Handle Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # CRITICAL: YIELD TO ASYNCIO LOOP
            await asyncio.sleep(0.01)
            
        cv2.destroyAllWindows()
        if video_out: video_out.release()
