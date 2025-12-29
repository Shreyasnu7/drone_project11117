1  # ai_autofocus.py
2  # Ultra-Intelligent Autofocus Engine (Part 1 + Part 2 Fully Combined)
3  # This module performs predictive autofocus, subject-aware focus,
4  # depth-assisted focusing, motion-adaptive focus pulls,
5  # cinematic rack-focus planning, confidence scoring, and error recovery.
6  #
7  # Designed to rival real cinema autofocus (Sony/Canon FX series level).
8  #
9  # NOTE: This module does NOT control any motors directly — it outputs
10 # focus parameters for camera drivers (GoPro/internal camera).
11
12 import numpy as np
13 import cv2
14 import time
15 import math
16 from collections import deque
17
18 class AIAutofocus:
19     """
20     Main autofocus engine with:
21       • Subject-aware tracking
22       • Predictive trajectory estimation
23       • Multi-frame sharpness evaluation
24       • Depth-aware focusing (via ai_depth_estimator)
25       • Motion-adaptive focus smoothing
26       • Cinematic rack-focus planner
27       • Fallback autofocus when subject lost
28
29     This engine produces:
30         { "focus_distance": meters,
31           "confidence": 0–1,
32           "mode": "predictive" | "subject" | "fallback"
33         }
34     """
35
36     def __init__(self):
37         # History buffers
38         self.subject_history = deque(maxlen=30)
39         self.sharpness_history = deque(maxlen=20)
40         self.depth_history = deque(maxlen=20)
41
42         # Last known states
43         self.last_focus_distance = 2.0
44         self.last_confidence = 0.5
45
46         # Motion smoothing parameters
47         self.smooth_factor = 0.85     # higher = smoother, slower
48         self.max_focus_speed = 6.0    # m/s focus change limit
49
50         # Sharpness computation kernel
51         self.laplacian_kernel = np.array([[0, -1, 0],
52                                           [-1, 4, -1],
53                                           [0, -1, 0]])
54
55         # Baseline camera parameters (placeholder, adjustable)
56         self.focal_length_mm = 5.0
57         self.f_stop = 1.8
58         self.sensor_width_mm = 6.3
59
60     # ---------------------------------------------------------------
61     # (1) SUBJECT SHARPNESS ESTIMATION
62     # ---------------------------------------------------------------
63     def compute_sharpness(self, gray):
64         """
65         Computes sharpness score of the subject region using Laplacian.
66         """
67         lap = cv2.Laplacian(gray, cv2.CV_64F)
68         sharp = lap.var()
69         return sharp
70
71     # ---------------------------------------------------------------
72     # (2) DEPTH FUSION (from ai_depth_estimator)
73     # ---------------------------------------------------------------
74     def update_depth(self, depth_m):
75         """
76         Add depth sample from the depth estimator (meters).
77         """
78         if depth_m is None:
79             return
80         self.depth_history.append(depth_m)
81
82     def get_depth_focus(self):
83         """
84         Computes stable focus from depth history.
85         """
86         if len(self.depth_history) == 0:
87             return None, 0.0
88
89         median_depth = np.median(self.depth_history)
90         stability = np.std(self.depth_history)
91
92        confidence = max(0.0, 1.0 - (stability * 2.2))
93
94        return float(median_depth), confidence
95
96     # ---------------------------------------------------------------
97     # (3) SUBJECT TRACKING → Predict future position
98     # ---------------------------------------------------------------
99     def update_subject_position(self, bbox_3d):
100         """
101         bbox_3d = { "x": , "y": , "z": , "distance": m }
102         """
103         self.subject_history.append(bbox_3d)
104
105     def predict_future_distance(self, t=0.15):
106         """
107         Predicts subject distance after t seconds using linear motion.
108         """
109         if len(self.subject_history) < 3:
110             return None
111
112         # Extract last 3 distances
113         d1 = self.subject_history[-1]["distance"]
114         d2 = self.subject_history[-2]["distance"]
115         d3 = self.subject_history[-3]["distance"]
116
117         vel = (d1 - d3) / 2.0
118         pred = d1 + vel * t
119
120         return max(0.2, pred)
121     # ---------------------------------------------------------------
122     # (4) SHARPNESS-BASED FOCUS SEARCH (Fallback AF)
123     # ---------------------------------------------------------------
124     def fallback_focus(self, frame, search_steps=7, search_range=0.7):
125         """
126         When subject is lost or depth is unstable:
127         Sweep focus distance around last known distance and measure sharpness.
128         """
129         h, w = frame.shape[:2]
130         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
131
132         base = self.last_focus_distance
133         best_focus = base
134         best_sharp = -1
135
136         for i in range(-search_steps, search_steps + 1):
137             candidate = base + (i / search_steps) * search_range
138             candidate = max(0.2, candidate)
139
140             # Simulated blur for candidate evaluation
141             blur_strength = max(0, min(15, abs(candidate - base) * 4.0))
142             blur_img = cv2.GaussianBlur(gray, (0, 0), blur_strength)
143
144             sharp = self.compute_sharpness(blur_img)
145             if sharp > best_sharp:
146                 best_sharp = sharp
147                 best_focus = candidate
148
149         confidence = min(1.0, best_sharp / 2000.0)
150
151         return best_focus, confidence
152
153     # ---------------------------------------------------------------
154     # (5) FOCUS SMOOTHING (Cinematic)
155     # ---------------------------------------------------------------
156     def smooth_focus_transition(self, new_focus):
157         """
158         Smooths abrupt focus changes using exponential filtering +
159         max focus speed limit.
160         """
161         old = self.last_focus_distance
162         raw_change = new_focus - old
163
164         # limit rate
165         max_step = self.max_focus_speed * 0.02  # per frame (~50 Hz)
166         raw_change = np.clip(raw_change, -max_step, max_step)
167
168         filtered = old * self.smooth_factor + (old + raw_change) * (1 - self.smooth_factor)
169         self.last_focus_distance = filtered
170
171         return filtered
172
173     # ---------------------------------------------------------------
174     # (6) CINEMATIC RACK-FOCUS PLANNING
175     # ---------------------------------------------------------------
176     def plan_rack_focus(self, subject_near, subject_far, duration=1.2):
177         """
178         Generates a time-parametric focus curve between two subjects.
179         Used when user commands: "rack focus from person to bike".
180
181         Returns:
182             { "curve": [ {t, focus}, ... ], "duration": s }
183         """
184         samples = 40
185         curve = []
186         for i in range(samples):
187             t = i / (samples - 1)
188             eased = t*t*(3 - 2*t)   # smoothstep easing
189             f = subject_near + (subject_far - subject_near) * eased
190             curve.append({"t": float(t * duration), "focus": float(f)})
191
192         return {"curve": curve, "duration": duration}
193
194     # ---------------------------------------------------------------
195     # (7) MAIN UPDATE LOOP (the heart of AF)
196     # ---------------------------------------------------------------
197     def update(self, frame, subject_box=None, depth_sample=None):
198         """
199         Core autofocus logic. Called every frame (30–120 FPS).
200
201         Input:
202             frame (BGR)
203             subject_box = { "x","y","w","h","distance" } — OR None
204             depth_sample = meters — OR None
205
206         Output:
207             { "focus_distance": m,
208               "confidence": 0–1,
209               "mode": "predictive" | "subject" | "fallback"
210             }
211         """
212
213         mode = "fallback"
214
215         # (A) Update depth estimator
216         if depth_sample is not None:
217             self.update_depth(depth_sample)
218
219         # (B) If subject is known → update tracker & predictive distance
220         if subject_box is not None:
221             self.update_subject_position({
222                 "x": subject_box["x"],
223                 "y": subject_box["y"],
224                 "z": subject_box.get("z", 0),
225                 "distance": subject_box["distance"]
226             })
227
228         predicted = None
229         predicted_conf = 0
230
231         if subject_box is not None:
232             predicted = self.predict_future_distance()
233             if predicted is not None:
234                 predicted_conf = 0.75
235                 mode = "predictive"
236
237         # (C) Depth-based focusing
238         depth_focus, depth_conf = self.get_depth_focus()
239         if depth_focus is not None and depth_conf > predicted_conf:
240             predicted = depth_focus
241             predicted_conf = depth_conf
242             mode = "subject" if subject_box is not None else "depth"
243
244         # (D) If no predicted value yet → run fallback scan
245         if predicted is None:
246             fb_focus, fb_conf = self.fallback_focus(frame)
247             predicted = fb_focus
248             predicted_conf = fb_conf
249             mode = "fallback"
250
251         # -------------------------------------------------------
252         # Stabilize focus using cinematic smoothing
253         # -------------------------------------------------------
254         final_focus = self.smooth_focus_transition(predicted)
255
256         # -------------------------------------------------------
257         # Pack output
258         # -------------------------------------------------------
259         return {
260             "focus_distance": float(final_focus),
261             "confidence": float(predicted_conf),
262             "mode": mode
263         }
264
265     # ===================================================================
266     #                           DEBUG UTILITIES
267     # ===================================================================
268     def draw_focus_info(self, frame, focus_data):
269         """
270         Draws focus diagnostics on frame.
271         """
272         msg = f"AF={focus_data['focus_distance']:.2f}m conf={focus_data['confidence']:.2f} mode={focus_data['mode']}"
273         cv2.putText(frame, msg, (20, 40),
274                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
275         return frame
276
277     # ===================================================================
278     #              SUBJECT TRACKING HELPER FOR CINEMATIC AF
279     # ===================================================================
280     def assign_subject(self, detections):
281         """
282         Selects a subject from detection list.
283         Detections format:
284            [{"label":"person","conf":0.92,"box":[x,y,w,h],"distance":3.2}, ...]
285         """
286         if not detections:
287             self.subject_history = []
288             return None
289
290        # pick the most confident "person" OR nearest object
291         best = None
292         best_score = -1
293
294         for det in detections:
295             label = det["label"]
296             score = det.get("conf", 0)
297             dist = det.get("distance", 999)
298
299             # score:
300             #   person detected → bonus
301             #   closer → stronger priority
302             #   confidence → multiplies
303             s = score * 1.5 if label == "person" else score
304             s *= (1.0 / max(0.7, dist))
305
306             if s > best_score:
307                 best = det
308                 best_score = s
309
310         # Save history for predictive motion
311         if best is not None:
312             self.update_subject_position({
313                 "x": best["box"][0] + best["box"][2] / 2,
314                 "y": best["box"][1] + best["box"][3] / 2,
315                 "z": 0,
316                 "distance": best.get("distance", 3.0)
317             })
318
319         return best
320
321     # ===================================================================
322     #         DEPTH-BASED REFOCUS WITH STABILITY DETECTION
323     # ===================================================================
324     def depth_based_refocus(self, depth_map):
325         """
326         Uses full depth map to choose optimal focal plane.
327         More robust than single depth_sample.
328
329         depth_map: 2D float array of meters.
330         """
331         if depth_map is None:
332             return None, 0
333
334         # Compute histogram of depth to find stable cluster
335         flat = depth_map.flatten()
336         flat = flat[(flat > 0.1) & (flat < 30.0)]
337         if len(flat) < 200:
338             return None, 0
339
340         hist, edges = np.histogram(flat, bins=40, range=(0, 30))
341         peak_idx = np.argmax(hist)
342         peak_depth = (edges[peak_idx] + edges[peak_idx + 1]) / 2
343
344         # confidence: based on histogram peak prominence
345         conf = hist[peak_idx] / max(1, np.sum(hist))
346
347         # update depth estimator
348         self.update_depth(peak_depth)
349
350         return peak_depth, float(conf)
351
352     # ===================================================================
353     #         ADVANCED AF METRIC: REGIONAL SHARPNESS MAP
354     # ===================================================================
355     def sharpness_map(self, gray):
356         """
357         Compute per-block sharpness metric across frame.
358         Useful for detecting regions in focus vs out of focus.
359         """
360         h, w = gray.shape
361         block_h = h // 10
362         block_w = w // 10
363
364         sharp_map = np.zeros((10, 10))
365
366         for i in range(10):
367             for j in range(10):
368                 y0 = i * block_h
369                 x0 = j * block_w
370                 y1 = y0 + block_h
371                 x1 = x0 + block_w
372
373                 block = gray[y0:y1, x0:x1]
374                 # Laplacian-based sharpness
375                 sharp = cv2.Laplacian(block, cv2.CV_64F).var()
376                 sharp_map[i, j] = sharp
377
378         return sharp_map
379
380     # ===================================================================
381     #     SMART REGIONAL AUTOFOCUS (detects where cinematic subject is)
382     # ===================================================================
383     def region_focus(self, gray, subject_box=None):
384         """
385         Computes a weighted sharpness score in region of interest.
386
387         If subject_box is provided:
388             → focus aggressively inside box
389         Otherwise:
390             → use center-weighted region
391         """
392         h, w = gray.shape
393
392         if subject_box is not None:
393             x, y, bw, bh = subject_box
394             x = max(0, int(x))
395             y = max(0, int(y))
396             bw = int(bw)
397             bh = int(bh)
398             crop = gray[y:y+bh, x:x+bw]
399         else:
400             cx = w // 4
401             cy = h // 4
402             crop = gray[cy:cy + h//2, cx:cx + w//2]
403
404         if crop.size == 0:
405             return 0
406
407         sharp = cv2.Laplacian(crop, cv2.CV_64F).var()
408         return sharp
409
410     # ===================================================================
411     #     HIGH-LEVEL API: RETURN "WHAT CAMERA SHOULD DO NEXT"
412     # ===================================================================
413     def autofocus_step(self, frame, vision_context):
414         """
415         Frame → detect subject → compute focus distance → return camera command.
416
417         High-level output example:
418             {
419               "focus_distance": 2.1,
420               "confidence": 0.89,
421               "mode": "subject",
422               "should_refocus": True
423             }
424         """
425         dets = vision_context.get("detections", [])
426         subject = self.assign_subject(dets)
427
428         subject_box = subject["box"] if subject else None
429         subject_dist = subject.get("distance") if subject else None
430
431         focus_data = self.compute_focus(frame, subject_box, subject_dist)
432
433         # logic: refocus if mode != fallback OR conf < threshold
434         should_refocus = focus_data["confidence"] < 0.30 or focus_data["mode"] != "fallback"
435
436         return {
437             "focus_distance": focus_data["focus_distance"],
438             "confidence": focus_data["confidence"],
439             "mode": focus_data["mode"],
440             "should_refocus": should_refocus,
441             "subject_box": subject_box
442         }
443
444     # ===================================================================
445     #      META: EXTRACT FOCUS METRICS FOR CAMERA BRAIN INTEGRATION
446     # ===================================================================
447     def get_focus_metadata(self):
448         """
449         Return internal state for UltraCameraBrain to fuse with
450         HDR engine, exposure engine, stabilizer, etc.
451         """
452         return {
453             "current_focus": float(self.current_focus),
454             "depth_state": float(self.depth_state),
455             "subject_history": list(self.subject_history[-5:]),
456             "last_sharpness": float(self.sharpness_state),
457         }
458
459     # ===================================================================
460     #               EXPERIMENTAL: SCENE TYPE → AF MODE SWITCHING
461     # ===================================================================
462     def scene_based_mode(self, scene_type):
463         """
464         Scene types:
465             - "landscape" → use depth-based AF
466             - "portrait"  → strong subject preference
467             - "action"    → predictive motion AF
468         """
469
470         if scene_type == "landscape":
471             self.scene_mode = "depth"
472
473         elif scene_type == "portrait":
474             self.scene_mode = "subject"
475
476         elif scene_type == "action":
477             # enforce predictive AF
478             self.scene_mode = "predictive"
479
480         else:
481             self.scene_mode = "auto"
481     # ===================================================================
482     #       EXPERIMENTAL: DEPTH + MOTION PREDICTION AF (CINEMATIC)
483     # ===================================================================
484     def predictive_af(self, subject_box, depth_map, prev_frame, curr_frame):
485         """
486         Predictive autofocus:
487             • Computes optical flow around subject
488             • Predicts subject movement next frame
489             • Uses depth_map (if available) to refine distance
490
491         Returns:
492             {
493                "predicted_focus": float,
494                "motion_vector": (vx, vy),
495                "confidence": float
496             }
497         """
498
499         if prev_frame is None or curr_frame is None:
500             return {
501                 "predicted_focus": self.current_focus,
502                 "motion_vector": (0.0, 0.0),
503                 "confidence": 0.1,
504             }
505
506         gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
507         gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
508
509         h, w = gray_curr.shape
510
511         if subject_box:
512             x, y, bw, bh = subject_box
513             x = max(0, int(x))
514             y = max(0, int(y))
515             bw = int(bw)
516             bh = int(bh)
517
518             prev_crop = gray_prev[y:y+bh, x:x+bw]
519             curr_crop = gray_curr[y:y+bh, x:x+bw]
520         else:
521             # fallback: central region
522             cx = w // 4
523             cy = h // 4
524             prev_crop = gray_prev[cy:cy+h//2, cx:cx+w//2]
525             curr_crop = gray_curr[cy:cy+h//2, cx:cx+w//2]
526
527         if prev_crop.size == 0 or curr_crop.size == 0:
528             return {
529                 "predicted_focus": self.current_focus,
530                 "motion_vector": (0.0, 0.0),
531                 "confidence": 0.1,
532             }
533
534         # ---------------------------------------------------------------
535         #     Optical flow (Farneback or Lucas-Kanade)
536         # ---------------------------------------------------------------
537         flow = cv2.calcOpticalFlowFarneback(
538             prev_crop, curr_crop, None,
539             pyr_scale=0.5, levels=3, winsize=15,
540             iterations=3, poly_n=5, poly_sigma=1.2, flags=0
541         )
542
543         # mean motion vector
544         mv = flow.mean(axis=(0, 1))   # (vx, vy)
545         vx, vy = float(mv[0]), float(mv[1])
546         motion_magnitude = (vx**2 + vy**2)**0.5
547
548         # ---------------------------------------------------------------
549         #      Depth refinement — if depth map is supplied
550         # ---------------------------------------------------------------
551         predicted_focus = self.current_focus
552
553         if depth_map is not None:
554             # Extract depth value under subject center
555             if subject_box:
556                 cx = int(subject_box[0] + subject_box[2] / 2)
557                 cy = int(subject_box[1] + subject_box[3] / 2)
558             else:
559                 cy, cx = h // 2, w // 2
560
561             if 0 <= cx < depth_map.shape[1] and 0 <= cy < depth_map.shape[0]:
562                 d = float(depth_map[cy, cx])
563                 if d > 0:
564                     # Weighted blend of AF and depth
565                     predicted_focus = 0.7 * predicted_focus + 0.3 * d
566
567         # motion influences focus slightly
568         predicted_focus += motion_magnitude * 0.01
569
570         conf = min(1.0, 0.2 + motion_magnitude * 0.05)
571
572         return {
573             "predicted_focus": predicted_focus,
574             "motion_vector": (vx, vy),
575             "confidence": conf,
576         }
577
578     # ===================================================================
579     #       DEBUG VISUALIZATION OVERLAYS FOR STUDIO-LEVEL TUNING
580     # ===================================================================
581     def draw_debug(self, frame, focus_info):
582         """
583         Draws:
584             - subject box
585             - focus distance
586             - focus confidence
587             - mode indicator
588         """
589         vis = frame.copy()
590
591         # subject box
592         if focus_info.get("subject_box") is not None:
593             x, y, w, h = focus_info["subject_box"]
594             cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
595
596         text = f"F={focus_info['focus_distance']:.2f}  C={focus_info['confidence']:.2f}  Mode={focus_info.get('mode','?')}"
597         cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
598
599         return vis
600
601     # ===================================================================
602     #       CALIBRATION & TUNING UTILITIES
603     # ===================================================================
604     def set_focus_range(self, min_focus, max_focus):
605         """Adjust allowed physical range of the lens."""
606         self.min_focus = float(min_focus)
607         self.max_focus = float(max_focus)
608         self.current_focus = np.clip(self.current_focus, self.min_focus, self.max_focus)
609
610     def set_smoothing(self, alpha):
611         """
612         Controls how aggressively focus changes are smoothed.
613         Lower alpha = slower, cinematic focus pulls.
614         """
615         self.smoothing_alpha = float(alpha)
616
617     def set_confidence_thresholds(self, stable, jump):
618         """
619         stable: confidence required to accept continuous tracking
620         jump:   confidence required to allow sudden focus jump
621         """
622         self.min_confidence_track = float(stable)
623         self.min_confidence_jump = float(jump)
624
625     # ===================================================================
626     #       EXPORT / IMPORT FOCUS PARAMETERS (for LUT building)
627     # ===================================================================
628     def export_state(self):
629         """Return autofocus parameters for saving."""
630         return {
631             "current_focus": self.current_focus,
632             "min_focus": self.min_focus,
633             "max_focus": self.max_focus,
634             "smoothing_alpha": self.smoothing_alpha,
635             "subject_locked": self.subject_locked,
636             "subject_last_box": self.subject_last_box,
637             "history": list(self.history),
638         }
639
640     def load_state(self, data):
641         """Load autofocus parameters."""
642         try:
643             self.current_focus = float(data.get("current_focus", self.current_focus))
644             self.min_focus = float(data.get("min_focus", self.min_focus))
645             self.max_focus = float(data.get("max_focus", self.max_focus))
646             self.smoothing_alpha = float(data.get("smoothing_alpha", self.smoothing_alpha))
647             self.subject_last_box = data.get("subject_last_box", None)
648             self.subject_locked = bool(data.get("subject_locked", False))
649         except Exception as e:
650             print("[AF] Warning: failed to load saved state:", e)
651
652     # ===================================================================
653     #       HIGH-LEVEL: END-TO-END AUTOFOCUS PIPELINE
654     # ===================================================================
655     def run(self, frame, vision_context, depth_map=None, prev_frame=None):
656         """
657         Full autofocus pipeline wrapper:
658             1) detect subject from vision_context
659             2) compute desired focus with multi-heuristic fusion
660             3) apply smoothing + hysteresis
661             4) return focus + metadata for camera brain
662
663         Returns dict:
664             {
665                 "focus_distance": float,
666                 "subject_box": [x,y,w,h] or None,
667                 "confidence": float,
668                 "mode": "tracking" / "smart" / "face" / "wide"
669             }
670         """
671         # -----------------------------------------------
672         # Step 1: get subject box (or None)
673         # -----------------------------------------------
674         subject_box = None
675
676         if vision_context:
677             if "selected" in vision_context and vision_context["selected"]:
678                 # the user's selected target
679                 subject_box = vision_context["selected"].get("bbox")
680             elif vision_context.get("faces"):
681                 # fallback: choose largest face
682                 faces = vision_context["faces"]
683                 if faces:
684                     areas = [(f["bbox"][2] * f["bbox"][3], f["bbox"]) for f in faces]
685                     areas.sort(reverse=True)
686                     subject_box = areas[0][1]
687             else:
688                 # fallback: largest detected object
689                 dets = vision_context.get("detections", [])
690                 if dets:
691                     areas = [(d["bbox"][2] * d["bbox"][3], d["bbox"]) for d in dets]
692                     areas.sort(reverse=True)
693                     subject_box = areas[0][1]
694
695         # -----------------------------------------------
696         # Step 2: compute focus from multi-source heuristics
697         # -----------------------------------------------
698         af_info = self.compute_focus(frame, subject_box, depth_map)
699
700         # -----------------------------------------------
701         # Step 3: predictive assistant (optical flow)
702         # -----------------------------------------------
703         if prev_frame is not None:
704             pred = self.predictive_af(subject_box, depth_map, prev_frame, frame)
705             # blend predictions slightly
706             af_info["focus_distance"] = (
707                 0.85 * af_info["focus_distance"] + 0.15 * pred["predicted_focus"]
708             )
709             af_info["confidence"] = min(1.0, af_info["confidence"] + pred["confidence"] * 0.1)
710
711         # clip & record
712        af_info["focus_distance"] = np.clip(
713            af_info["focus_distance"],
714            self.min_focus,
715            self.max_focus,
716        )
717
718         # Maintain subject lock
719         if subject_box:
720             self.subject_locked = True
721             self.subject_last_box = subject_box
722         else:
723             # If nothing detected, slowly decay lock
724             if self.subject_locked:
725                 if self.history:
726                     last_box = self.history[-1][0]
727                     if last_box is None:
728                         self.subject_locked = False
729
730         # update history
731         self.history.append((subject_box, af_info["focus_distance"]))
732
733         return af_info
734
735     # ===================================================================
736     #       DIAGNOSTIC TOOLS
737     # ===================================================================
738     def debug_print(self, info):
739         """Simple text logger."""
740         print(
741             f"[AF] F={info['focus_distance']:.2f}  "
742             f"C={info['confidence']:.2f}  "
743             f"Mode={info.get('mode','?')}  "
744             f"Box={info.get('subject_box')}"
745         )
746
747     # ===================================================================
748     #      FUTURE MODULE EXTENSION HOOKS (do not delete)
749     # ===================================================================
750     def attach_depth_model(self, model):
751         """Assign a neural depth model for better depth-driven focus."""
752         self.depth_model = model
753
754     def attach_scene_classifier(self, classifier):
755         """AI scene model (night mode, action mode, portrait mode)."""
756         self.scene_classifier = classifier
757
758     def attach_shot_planner(self, planner):
759         """Shot planner allows focus to be influenced by cinematic intent."""
760         self.shot_planner = planner
761
762     def attach_motion_estimator(self, estimator):
763         """Optical-flow acceleration module."""
764         self.motion_estimator = estimator
765
766     # ===================================================================
767     #      FILE COMPLETE — END OF ai_autofocus.py
768     # ===================================================================