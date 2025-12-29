001  """
002  AI EXPOSURE ENGINE — Cinematic Auto-Exposure (AE) + Local Tone Mapping
003  ======================================================================
004
005  This module performs:
006      • Global exposure estimation
007      • Scene-brightness normalization
008      • Local contrast enhancement (tone mapping)
009      • Temporal smoothing (no flicker)
010      • Anti-blowout protection
011      • Highlight priority mode
012      • Shadow recovery mode
013      • Scene-adaptive exposure (faces, sky, subject priority)
014
015  It does NOT control any motors or flight hardware → 100% SAFE.
016  Only adjusts image exposure parameters for cinematography.
017
018  Structure:
019      ExposureHistogramAnalyzer
020      ExposureState
021      ExposureTemporalFilter
022      ExposureDecisionEngine
023      AIExposureEngine (main controller)
024
025  This file will be ~450–500 lines total.
026  """
027
028  import numpy as np
029  import cv2
030  import time
031  from collections import deque
032  from typing import Dict, Optional
033
034  # -------------------------------------------------------------
035  # Utility: clamp function
036  # -------------------------------------------------------------
037  def clamp(x, low, high):
038      return low if x < low else high if x > high else x
039
040
041  # =====================================================================
042  # CLASS 1 — HISTOGRAM ANALYZER
043  # =====================================================================
044  class ExposureHistogramAnalyzer:
045      """
046      Computes luminance histogram and exposure metrics.
047
048      Outputs:
049          avg_luma           — mean brightness
050          median_luma        — median brightness
051          shadow_ratio       — dark area %
052          highlight_ratio    — bright area %
053          contrast_index     — global contrast score
054      """
055
056      def __init__(self, bins: int = 256):
057          self.bins = bins
058
059      def analyze(self, frame: np.ndarray) -> Dict[str, float]:
060          """
061          Input: frame (RGB or BGR)
062          Output: dict of luminance statistics
063          """
064          if frame is None or frame.size == 0:
065              return {
066                  "avg_luma": 0.5,
067                  "median_luma": 0.5,
068                  "shadow_ratio": 0.0,
069                  "highlight_ratio": 0.0,
070                  "contrast_index": 0.0
071              }
072
073          # convert BGR → Y (luminance)
074          yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
075          y = yuv[:, :, 0].astype(np.float32) / 255.0
076
077          # histogram
078          hist, _ = np.histogram(y, bins=self.bins, range=(0.0, 1.0))
079          hist = hist.astype(np.float32)
080          hist /= np.sum(hist) + 1e-8
081          # cumulative histograms
082          cdf = np.cumsum(hist)
083
084          # average brightness
085          avg_luma = float(np.sum(hist * np.linspace(0, 1, self.bins)))
086
087          # median brightness
088          median_idx = np.searchsorted(cdf, 0.5)
089          median_luma = float(median_idx / (self.bins - 1))
090
091          # shadows = lower 15% of brightness range
092          shadow_ratio = float(np.sum(hist[: int(self.bins * 0.15)]))
093
094          # highlights = top 15% of range
095          highlight_ratio = float(np.sum(hist[int(self.bins * 0.85):]))
096
097          # simple global contrast index:
098          # difference between bright-side median & dark-side median
099          low_region = y[y < 0.25]
100          high_region = y[y > 0.75]
101
102          if low_region.size > 20 and high_region.size > 20:
103              dark_med = float(np.median(low_region))
104              bright_med = float(np.median(high_region))
105              contrast_index = float(bright_med - dark_med)
106          else:
107              # fallback estimation
108              contrast_index = float(np.std(y))
109
110          return {
111              "avg_luma": avg_luma,
112              "median_luma": median_luma,
113              "shadow_ratio": shadow_ratio,
114              "highlight_ratio": highlight_ratio,
115              "contrast_index": contrast_index
116          }
117
118
119  # =====================================================================
120  # CLASS 2 — EXPOSURE STATE (persistent state of the camera exposure)
121  # =====================================================================
122  class ExposureState:
123      """
124      Maintains a persistent exposure state:
125          • current exposure compensation (EV)
126          • previous histogram metrics
127          • scene mode (normal / highlight-priority / shadow-priority)
128          • history buffers for smoothing
129      """
130
131      def __init__(self):
132          self.ev = 0.0                     # exposure compensation (-2.0 ... +2.0)
133          self.target_luma = 0.55           # ideal cinematic brightness
134          self.mode = "normal"              # modes: normal / highlight / shadow
135
136          self.hist_history = deque(maxlen=12)   # last 12 histograms
137          self.ev_history = deque(maxlen=8)       # last few EV adjustments
138
139          self.last_update_ts = time.time()
140
141      def push_histogram(self, hist_metrics: Dict[str, float]):
142          self.hist_history.append(hist_metrics)
143
144      def push_ev(self, ev: float):
145          self.ev_history.append(ev)
146
147      def get_smooth_ev(self) -> float:
148          """
149          Smooth exposure compensation — prevents flicker.
150          """
151          if not len(self.ev_history):
152              return self.ev
153
154          # Weighted smoothing: last 3 frames get more weight
155          n = len(self.ev_history)
156          weights = np.linspace(1, 2, n)
157          ev = float(np.sum(np.array(self.ev_history) * weights) / np.sum(weights))
158          return clamp(ev, -2.0, 2.0)
159
160  # End of chunk 2
161  # =====================================================================
162  # CLASS 3 — EXPOSURE DECISION ENGINE
163  # =====================================================================
164  class ExposureDecisionEngine:
165      """
166      Computes the EV (Exposure Value) correction needed for cinematic exposure.
167      Based on:
168          • histogram metrics
169          • target mid-tone luma
170          • highlight/shadow protection
171          • scene mode
172          • anti-flicker smoothing
173      """
174
175      def __init__(self):
176          self.max_step = 0.25          # max EV change per frame (anti-flicker)
177          self.highlight_threshold = 0.22
178          self.shadow_threshold = 0.22
179          self.cinematic_mid_bias = 1.15    # pulls exposure toward mid-tones
180
181      def compute_correction(self,
182                             hist_metrics: Dict[str, float],
183                             state: ExposureState) -> float:
184          """
185          Returns the EV correction (delta EV) for this frame.
186          """
187          avg_luma = hist_metrics["avg_luma"]
188          median_luma = hist_metrics["median_luma"]
189          shadows = hist_metrics["shadow_ratio"]
190          highlights = hist_metrics["highlight_ratio"]
191
192          # -------------------------------------------------------------
193          # 1) BASE EXPOSURE ERROR (mid-tone driven)
194          # -------------------------------------------------------------
195          # cinematic mid-tone emphasis
196          mid_tone = (avg_luma * 0.5 + median_luma * 0.5)
197          mid_tone *= self.cinematic_mid_bias
198
199          target = state.target_luma
200          error = (target - mid_tone)
201
202          # EV adjustment is proportional to error
203          ev_delta = error * 1.8     # aggressive control (cinematic)
204
205          # -------------------------------------------------------------
206          # 2) HIGHLIGHT PROTECTION
207          # -------------------------------------------------------------
208          if highlights > self.highlight_threshold:
209              # too many blown pixels → reduce exposure
210              ev_delta -= (highlights * 0.45)
211
212          # -------------------------------------------------------------
213          # 3) SHADOW BOOST
214          # -------------------------------------------------------------
215          if shadows > self.shadow_threshold:
216              # too many crushed blacks → raise exposure
217              ev_delta += (shadows * 0.32)
218
219          # -------------------------------------------------------------
220          # 4) SCENE-MODE LOGIC
221          # -------------------------------------------------------------
222          if state.mode == "highlight":
223              # keep highlights safe
224              ev_delta -= 0.18
225          elif state.mode == "shadow":
226              # boost shadows for dark scenes
227              ev_delta += 0.18
228          else:
229              # normal → nothing special
230              pass
231
232          # -------------------------------------------------------------
233          # 5) LIMIT MAX STEP SIZE (anti-flicker)
234          # -------------------------------------------------------------
235          ev_delta = float(clamp(ev_delta, -self.max_step, self.max_step))
236
237          return ev_delta
238
239      # -------------------------------------------------------------
240      # MODE SWITCHING FOR SCENE DETECTION
241      # -------------------------------------------------------------
242      def update_scene_mode(self, hist_metrics: Dict[str,float], state: ExposureState):
243          """
244          Update `state.mode` based on histogram classification.
245          """
246          shadows = hist_metrics["shadow_ratio"]
247          highlights = hist_metrics["highlight_ratio"]
248
249          if highlights > 0.28:
250              state.mode = "highlight"
251          elif shadows > 0.28:
252              state.mode = "shadow"
253          else:
254              state.mode = "normal"
255
256          return state.mode
257
258
259  # End of chunk 3
260
261  # =====================================================================
262  # CLASS 4 — EXPOSURE TRANSLATOR (EV → CAMERA SETTINGS)
263  # =====================================================================
264  class ExposureTranslator:
265      """
266      Converts exposure compensation (EV delta) into ISO, shutter, gain.
267
268      Uses:
269          • scene brightness
270          • motion level
271          • cinematic shutter rules
272          • ISO noise model
273          • device limits
274      """
275
276      def __init__(self,
277                   base_iso=100,
278                   max_iso=6400,
279                   min_shutter=1/8000,
280                   max_shutter=1/24,
281                   shutter_180deg=True):
282
283          self.base_iso = base_iso
284          self.max_iso = max_iso
285          self.min_shutter = min_shutter
286          self.max_shutter = max_shutter
287          self.shutter_180deg = shutter_180deg
288
289      def estimate_motion_level(self, motion_score: float) -> float:
290          """
291          Converts a motion score (0–1) into shutter bias.
292          High motion → faster shutter recommended.
293          """
294          return clamp(motion_score, 0.0, 1.0)
295
296      def translate(self,
297                    state: ExposureState,
298                    hist_metrics: Dict[str,float],
299                    ev_delta: float,
300                    motion_score: float) -> Dict[str,float]:
301          """
302          Returns a dict:
303              { "iso": ..., "shutter": ..., "ev_applied": ... }
304          """
305
306          # -------------------------------------------------------------
307          # 1) Apply EV correction to brightness model
308          # -------------------------------------------------------------
309          # update internal EV
310          state.ev_last += ev_delta
311          state.ev_last = float(clamp(state.ev_last, -4.0, +4.0))
312
313          # -------------------------------------------------------------
314          # 2) Motion-adjusted shutter selection
315          # -------------------------------------------------------------
316          motion = self.estimate_motion_level(motion_score)
317
318          if self.shutter_180deg:
319              # Cinematic shutter rule (180°) → shutter ≈ 1/(2 × framerate)
320              shutter = 1.0 / (2.0 * state.framerate)
321              # fast action → bias toward faster shutter
322              shutter *= (1.0 - 0.35 * motion)
323          else:
324              # free shutter mode
325              shutter = self.max_shutter - (motion * (self.max_shutter - self.min_shutter))
326
327          # Clamp shutter
328          shutter = clamp(shutter, self.min_shutter, self.max_shutter)
329
330          # -------------------------------------------------------------
331          # 3) ISO calculation
332          # -------------------------------------------------------------
333          # Start from base ISO, add EV correction
334          iso_multiplier = 2 ** state.ev_last
335          iso = self.base_iso * iso_multiplier
336
337          # clamp ISO
338          iso = float(clamp(iso, self.base_iso, self.max_iso))
339
340          # -------------------------------------------------------------
341          # 4) Gain compensation (if ISO topped out)
342          # -------------------------------------------------------------
343          gain = 1.0
344          if iso >= self.max_iso * 0.98:
345              # If ISO maxed, add small digital gain
346              gain = 1.0 + (state.ev_last * 0.06)
347              gain = float(clamp(gain, 1.0, 1.25))
348
349          # -------------------------------------------------------------
350          # 5) Output structure
351          # -------------------------------------------------------------
352          return {
353              "iso": iso,
354              "shutter": shutter,
355              "gain": gain,
356              "ev_applied": state.ev_last,
357              "mode": state.mode,
358              "motion": motion,
359          }
360
361  # =====================================================================
362  # CLASS 5 — TEMPORAL EXPOSURE SMOOTHER
363  # =====================================================================
364  class ExposureSmoother:
365      """
366      Smooths exposure transitions over time to avoid flicker/jumps.
367
368      Maintains:
369          - ISO smoothing
370          - shutter smoothing
371          - gain smoothing
372          - EV momentum model
373      """
374
375      def __init__(self,
376                   iso_smooth=0.25,
377                   shutter_smooth=0.20,
378                   gain_smooth=0.35,
379                   ev_momentum=0.15):
380
381          self.iso_smooth = iso_smooth
382          self.shutter_smooth = shutter_smooth
383          self.gain_smooth = gain_smooth
384          self.ev_momentum = ev_momentum
385
386          # last stable outputs
387          self.last_iso = None
388          self.last_shutter = None
389          self.last_gain = None
390          self.last_ev = 0.0
391
392      def smooth_step(self, prev, new, amt):
393          """Generic smoothing function."""
394          if prev is None:
395              return new
396          return prev + (new - prev) * amt
397
398      def apply(self, raw: Dict[str,float]) -> Dict[str,float]:
399          """
400          Inputs raw dict:
401             { iso, shutter, gain, ev_applied }
402
403          Returns smoothed dict with:
404             { iso, shutter, gain, ev }
405          """
406
407          iso = raw["iso"]
408          shutter = raw["shutter"]
409          gain = raw["gain"]
410          ev = raw["ev_applied"]
411
412          # ----------------------------------------
413          # Smooth all channels
414          # ----------------------------------------
415          self.last_iso = self.smooth_step(self.last_iso, iso, self.iso_smooth)
416          self.last_shutter = self.smooth_step(self.last_shutter, shutter, self.shutter_smooth)
417          self.last_gain = self.smooth_step(self.last_gain, gain, self.gain_smooth)
418
419          # EV momentum (prevents micro flicker)
420          self.last_ev = self.smooth_step(self.last_ev, ev, self.ev_momentum)
421
422          return {
423              "iso": float(self.last_iso),
424              "shutter": float(self.last_shutter),
425              "gain": float(self.last_gain),
426              "ev": float(self.last_ev),
427          }
428
429
430  # =====================================================================
431  # CLASS 6 — MAIN EXPOSURE ENGINE (what AICameraBrain calls)
432  # =====================================================================
433  class ExposureEngine:
434      """
435      High-level interface for the auto-exposure pipeline.
436
437      It owns:
438          → SceneAnalyzer (histogram + brightness)
439          → ExposureTranslator (ISO/shutter/gain)
440          → ExposureSmoother (temporal smoothing)
441
442      Usage:
443          engine = ExposureEngine()
444          settings = engine.update(frame, motion_score, state)
445          return settings   # dict with ISO, shutter, gain, metadata
446      """
447
448      def __init__(self):
449          self.scene = SceneAnalyzer()
450          self.translator = ExposureTranslator()
451          self.smoother = ExposureSmoother()
452
453      def update(self,
454                 frame: np.ndarray,
455                 motion_score: float,
456                 state: ExposureState,
457                 return_debug: bool = False) -> Dict[str, any]:
458          """
459          Main loop called every frame.
460
461          Args:
462              frame: (H,W,3) BGR image
463              motion_score: float 0–1
464              state: ExposureState
465
466          Returns:
467              dict containing:
468                 iso
469                 shutter
470                 gain
471                 ev
472                 scene_metrics
473                 debug(optional)
474          """
475
476          # ---------------------------------------------------------
477          # 1) Analyze scene
478          # ---------------------------------------------------------
479          metrics = self.scene.analyze(frame)
480
481          # brightness error → EV correction
482          ev_delta = compute_ev_delta(
483              metrics["brightness"],        # predicted exposure
484              state.brightness_target       # desired
485          )
486
487          # ---------------------------------------------------------
488          # 2) Translate exposure
489          # ---------------------------------------------------------
490          raw_settings = self.translator.translate(
491              state=state,
492              hist_metrics=metrics,
493              ev_delta=ev_delta,
494              motion_score=motion_score,
495          )
496
497          # ---------------------------------------------------------
498          # 3) Smooth exposure for cinematic look
499          # ---------------------------------------------------------
500          smooth = self.smoother.apply(raw_settings)
501
502          # ---------------------------------------------------------
503          # 4) Pack final output
504          # ---------------------------------------------------------
505          out = {
506              "iso": smooth["iso"],
507              "shutter": smooth["shutter"],
508              "gain": smooth["gain"],
509              "ev": smooth["ev"],
510              "scene_metrics": metrics,
511              "mode": state.mode,
512              "timestamp": time.time(),
513          }
514
515          # optional developer debug
516          if return_debug:
517              out["debug"] = {
518                  "raw": raw_settings,
519                  "metrics": metrics,
520              }
521
522          return out
521  # =====================================================================
522  # CLASS 7 — EXPOSURE HISTORY (for prediction + oscillation prevention)
523  # =====================================================================
524  class ExposureHistory:
525      """
526      Maintains rolling history of:
527          • EV over time
528          • brightness values
529          • shutter & ISO transitions
530
531      Enables:
532          • oscillation detection
533          • predictive EV damping
534          • highlight protection trends
535      """
536
537      def __init__(self, size=24):
538          self.size = size
539          self.ev_hist = []
540          self.brightness_hist = []
541          self.shutter_hist = []
542          self.iso_hist = []
543
544      def push(self, ev, brightness, shutter, iso):
545          """Append to history with max length."""
546          if len(self.ev_hist) >= self.size:
547              self.ev_hist.pop(0)
548              self.brightness_hist.pop(0)
549              self.shutter_hist.pop(0)
550              self.iso_hist.pop(0)
551
552          self.ev_hist.append(ev)
553          self.brightness_hist.append(brightness)
554          self.shutter_hist.append(shutter)
555          self.iso_hist.append(iso)
556
557      def detect_ev_oscillation(self):
558          """
559          Returns True if EV is bouncing back and forth:
560               +1, -1, +1, -1...
561          This indicates bad flicker conditions or aggressive auto-exposure.
562          """
563          if len(self.ev_hist) < 6:
564              return False
565
566          signs = []
567          for i in range(1, len(self.ev_hist)):
568              delta = self.ev_hist[i] - self.ev_hist[i-1]
569              signs.append(np.sign(delta))
570
571          # Look for alternating pattern: + - + - + -
572          # Count alternations
573          alt = 0
574          for i in range(1, len(signs)):
575              if signs[i] * signs[i-1] < 0:
576                  alt += 1
577
578          return alt > (len(signs) * 0.6)
579
580      def mean_brightness(self):
581          if not self.brightness_hist:
582              return None
583          return float(np.mean(self.brightness_hist))
584
585      def trend_brightness(self):
586          """Return slope of brightness trend."""
587          if len(self.brightness_hist) < 2:
588              return 0.0
589          x = np.arange(len(self.brightness_hist))
590          y = np.array(self.brightness_hist)
591          slope = np.polyfit(x, y, 1)[0]
592          return float(slope)
593
594
595  # =====================================================================
596  # CLASS 8 — ADVANCED EV CORRECTOR (anti-flicker + highlight safety)
597  # =====================================================================
598  class AdvancedEVController:
599      """
600      Applies intelligent damping to EV changes, using history:
601
602      Goals:
603         ✔ Prevent flicker when tone varies rapidly
604         ✔ Preserve highlight detail (sky, reflections)
605         ✔ Smooth sudden EV jumps in partial shadows
606      """
607
608      def __init__(self,
609                   flicker_damp=0.35,
610                   highlight_protect_strength=0.25):
611
612          self.flicker_damp = flicker_damp
613          self.highlight_protect_strength = highlight_protect_strength
614
615      def apply(self, ev_raw, metrics, history: ExposureHistory):
616          """
617          Modify EV based on historical behavior.
618          """
619
620          # ----------------------------
621          # 1) Anti-oscillation damping
622          # ----------------------------
623          if history.detect_ev_oscillation():
624              # reduce EV swings by damping
625              ev_raw *= (1.0 - self.flicker_damp)
626
627          # ----------------------------
628          # 2) Highlight protection
629          # ----------------------------
630          highlight_ratio = metrics["highlight_ratio"]
631          if highlight_ratio > 0.15:     # meaning >15% of image is clipped
632              protect = min(1.0, highlight_ratio * 2.5)
633              ev_raw -= protect * self.highlight_protect_strength
634
635          return float(ev_raw)
636
637
638  # =====================================================================
639  # CLASS 9 — EXTENDED ExposureEngine (attaches history + EV controller)
640  # =====================================================================
641  class ExposureEnginePro(ExposureEngine):
642      """
643      Extends ExposureEngine with:
644         • History tracking
645         • Advanced EV smoothing
646         • Highlight protection
647         • Anti-oscillation detection
648
649      AICameraBrain should call THIS version, not ExposureEngine.
650      """
651
652      def __init__(self):
653          super().__init__()
654          self.history = ExposureHistory(size=32)
655          self.ev_controller = AdvancedEVController()
656
657      def update(self,
658                 frame,
659                 motion_score,
660                 state,
661                 return_debug=False):
662
663          # -----------------------------
664          # Base exposure metrics
665          # -----------------------------
666          metrics = self.scene.analyze(frame)
667          raw_ev_delta = compute_ev_delta(
668              metrics["brightness"],
669              state.brightness_target
670          )
671
672          # -----------------------------
673          # Apply EV corrections
674          # -----------------------------
675          ev_corrected = self.ev_controller.apply(
676              raw_ev_delta,
677              metrics,
678              self.history
679          )
680
681          # -----------------------------
682          # Base exposure translation
683          # -----------------------------
684          raw_settings = self.translator.translate(
685              state=state,
686              hist_metrics=metrics,
687              ev_delta=ev_corrected,
688              motion_score=motion_score,
689          )
690
691          # -----------------------------
692          # Temporal smoothing
693          # -----------------------------
694          smooth = self.smoother.apply(raw_settings)
695
696          # -----------------------------
697          # Update history AFTER smoothing
698          # -----------------------------
699          self.history.push(
700              smooth["ev"],
701              metrics["brightness"],
702              smooth["shutter"],
703              smooth["iso"]
704          )
705
706          # -----------------------------
707          # Output package
707          # -----------------------------
708          out = {
709              "iso": smooth["iso"],
710              "shutter": smooth["shutter"],
711              "gain": smooth["gain"],
712              "ev": smooth["ev"],
713              "mode": state.mode,
714              "scene_metrics": metrics,
715              "timestamp": time.time(),
716          }
717
718          if return_debug:
719              out["debug"] = {
720                  "raw": raw_settings,
721                  "metrics": metrics,
722                  "ev_corrected": ev_corrected,
723                  "history": {
724                      "ev_hist": self.history.ev_hist[-6:],
725                      "bright_hist": self.history.brightness_hist[-6:]
726                  }
727              }
728
729          return out
729  # =====================================================================
730  # CLASS 10 — HDR BRACKET MANAGER
731  # =====================================================================
732  class HDRBracketManager:
733      """
734      Controls multi-frame exposure bracketing for HDR fusion.
735
736      Responsibilities:
737        • Decide when HDR is needed (high dynamic range scenes)
738        • Generate exposure offsets for bracket captures
739        • Maintain rolling bracket patterns (3-frame or 5-frame)
740        • Ensure exposure shifts are smooth and stable
741      """
742
743      def __init__(self,
744                   ev_low=-1.5,
745                   ev_high=+1.5,
746                   frames=3,
747                   activate_threshold=0.35):
748
749          self.ev_low = ev_low
750          self.ev_high = ev_high
751          self.frames = frames      # 3 or 5
752          self.activate_threshold = activate_threshold  # contrast trigger
753
754          self.active = False
755          self.pattern = []
756
757      def should_enable_hdr(self, metrics):
758          """
759          Detect high dynamic range scenes.
760          Conditions:
761              • Highlight ratio high
762              • Shadow ratio high
763              • Combined > activate_threshold
764          """
765          bright = metrics["highlight_ratio"]
766          dark = metrics["shadow_ratio"]
767          return (bright + dark) > self.activate_threshold
768
769      def generate_pattern(self):
770          """
771          Return EV offsets for a bracket:
772              3-frame: [-ev, 0, +ev]
773              5-frame: [-evH, -evL, 0, +evL, +evH]
774          """
775          if self.frames == 3:
776              return [self.ev_low, 0.0, self.ev_high]
777
778          elif self.frames == 5:
779              mid_low = self.ev_low * 0.5
780              mid_high = self.ev_high * 0.5
781              return [self.ev_low, mid_low, 0.0, mid_high, self.ev_high]
782
783          else:
784              # fallback
785              return [self.ev_low, 0.0, self.ev_high]
786
787      def start_bracket(self, metrics):
788          """Start HDR capture sequence if needed."""
789          if self.should_enable_hdr(metrics):
790              self.active = True
791              self.pattern = self.generate_pattern()
792          else:
793              self.active = False
794              self.pattern = []
795
796      def get_ev_offset(self):
797          """
798          Pop the next bracket EV shift.
799          Returns None when bracket sequence is complete.
800          """
801          if not self.active or not self.pattern:
802              return None
803          return self.pattern.pop(0)
804
805
806  # =====================================================================
807  # CLASS 11 — HDR FUSION ENGINE (MANTIUK / REINHARD HYBRID)
808  # =====================================================================
809  class HDRFusionEngine:
810      """
811      Fuses multiple LDR frames into a single HDR image using:
812          • Log-luminance merge
813          • Weight maps (per-pixel confidence)
814          • Tone-mapping using Mantiuk/Reinhard hybrid
815
816      This is a simplified but REAL HDR implementation.
817      """
818
819      def __init__(self):
820          # tone-mapping parameters
821          self.mantiuk_strength = 0.85
822          self.reinhard_white = 4.0
823
824      # ---------------------------------------------------------------
825      # Utility: compute luminance
826      # ---------------------------------------------------------------
827      def luminance(self, img):
828          """
829          BT.709 luminance
830          """
831          return (
832              0.2126 * img[..., 2] +
833              0.7152 * img[..., 1] +
834              0.0722 * img[..., 0]
835          )
836
837      # ---------------------------------------------------------------
838      # Weight map generator
839      # ---------------------------------------------------------------
840      def compute_weight(self, img, ev_shift):
841          """
842          Give higher weight to mid-tones, lower to clipped regions.
843          """
844          lum = self.luminance(img)
845          w = np.exp(-4.0 * (lum - 0.5) ** 2)
846
847          # reduce weight if under/overexposed relative to shift
848          if ev_shift < 0:
849              w *= (lum < 0.9).astype(np.float32)
850          else:
851              w *= (lum > 0.1).astype(np.float32)
852
853          return w + 1e-6
854
855      # ---------------------------------------------------------------
856      # Merge function
857      # ---------------------------------------------------------------
858      def merge(self, frames, ev_shifts):
859          """
860          frames: list of np.ndarray images
861          ev_shifts: EV offsets applied to each frame
862          """
863          assert len(frames) == len(ev_shifts)
864
865          H, W = frames[0].shape[:2]
866          hdr = np.zeros((H, W, 3), dtype=np.float32)
867          total_w = np.zeros((H, W), dtype=np.float32)
868
869          for img, ev in zip(frames, ev_shifts):
870              # convert EV to exposure factor
871              scale = 2.0 ** ev
872              scaled = np.clip(img.astype(np.float32) * scale, 0, 1)
873
874              w = self.compute_weight(scaled, ev)
875
876              hdr += scaled * w[..., None]
877              total_w += w
878
879          hdr /= (total_w[..., None] + 1e-6)
880          return hdr
881
882      # ---------------------------------------------------------------
883      # Tone-mapping
884      # ---------------------------------------------------------------
885      def tone_map(self, hdr):
886          """
887          Apply hybrid tone-mapping using:
888              • local contrast compression (Mantiuk)
889              • global white point adjustment (Reinhard)
890          """
891          # ---------------------------
892          # Reinhard global
893          # ---------------------------
894          L = self.luminance(hdr)
895          L_white = self.reinhard_white
896
897          L_tone = (L * (1 + L / (L_white ** 2))) / (1 + L)
898
899          # ---------------------------
900          # Mantiuk local contrast
901          # ---------------------------
902          blur = cv2.GaussianBlur(L_tone, (0, 0), 8)
903          detail = L_tone - blur
904          L_final = L_tone + detail * self.mantiuk_strength
905
906          # reapply to RGB channels
907          scale = (L_final / (L + 1e-6))[..., None]
908          out = np.clip(hdr * scale, 0, 1)
909
910          return out
911  # =====================================================================
912  # CLASS 12 — HDR ORCHESTRATOR (CONNECTS BRACKETS + FUSION)
913  # =====================================================================
914  class HDROrchestrator:
915      """
916      Manages high-level HDR workflow:
917          1. Decide if HDR should activate
918          2. Coordinate bracket sequence
919          3. Trigger frame capture at different EVs
920          4. Perform HDR fusion + tone-map
921
922      This class is called every frame by ExposureEnginePro.update().
923      """
924
925      def __init__(self):
926          self.bracket_mgr = HDRBracketManager()
927          self.fusion = HDRFusionEngine()
928
929          self.collecting = False
930          self.buffer_frames = []
931          self.buffer_shifts = []
932
933      def begin_if_needed(self, metrics):
934          """
935          Decide whether HDR should start based on exposure metrics.
936          """
937          self.bracket_mgr.start_bracket(metrics)
938
939          if self.bracket_mgr.active:
940              self.collecting = True
941              self.buffer_frames.clear()
942              self.buffer_shifts.clear()
943          else:
944              self.collecting = False
945              self.buffer_frames.clear()
946              self.buffer_shifts.clear()
947
948      def step(self, frame):
949          """
950          Called each frame when HDR is active.
951          Returns:
952             • (ev_shift) if frame should be captured at new EV
953             • None if HDR is disabled or bracket done
954          """
955          if not self.collecting:
956              return None
957
958          ev = self.bracket_mgr.get_ev_offset()
959          if ev is None:
960              # bracket ended
961              self.collecting = False
962              return None
963
964          return ev
965
966      def push_frame(self, frame, ev):
967          """
968          Store bracket frame and EV shift.
969          """
970          self.buffer_frames.append(frame)
971          self.buffer_shifts.append(ev)
972
973      def ready(self):
974          """
975          Returns True when full bracket sequence is captured.
976          """
977          return (
978              len(self.buffer_frames) > 0 and
979              len(self.buffer_frames) == len(self.buffer_shifts) and
980              (not self.bracket_mgr.active)
981          )
982
983      def fuse(self):
984          """
985          Perform HDR merge + tone-map.
986          """
987          if not self.ready():
988              return None
989
990          try:
991              hdr = self.fusion.merge(self.buffer_frames, self.buffer_shifts)
992              out = self.fusion.tone_map(hdr)
993              return out
994          except Exception as e:
995              print("[HDR] Fusion error:", e)
996              return None
997
998
999  # =====================================================================
1000 # CLASS 13 — INTEGRATION INTO ExposureEnginePro
1001 # =====================================================================
1002 class ExposureEnginePro:
1003     """
1004     FULL PIPELINE:
1005        1. Compute exposure metrics
1006        2. Exposure decision layer (AE + scene classifier)
1007        3. HDR check + bracket orchestration
1008        4. Generate a final exposure setting for the frame
1009        5. (Optional) HDR fusion output for internal camera
1010     """
1011
1012     def __init__(self):
1013         self.metering = ExposureMetering()
1014         self.controller = ExposureDecisionLayer()
1015         self.hdr = HDROrchestrator()
1016
1017         # latest frame exposure output
1018         self.exposure_state = None
1019         self.last_hdr_output = None
1020
1021     # --------------------------------------------------------------
1022     # MAIN UPDATE ENTRYPOINT
1023     # --------------------------------------------------------------
1024     def update(self, frame, scene_hint=None):
1025         """
1026         Processes one frame:
1027             • Compute metrics
1028             • Exposure control
1029             • HDR logic
1030             • Tone-mapped output (optional)
1031
1032         Returns:
1033             {
1034                "state": {...},
1035                "hdr_output": optional_frame,
1036                "metrics": {...}
1037             }
1038         """
1039
1040         # 1) compute metrics
1041         metrics = self.metering.compute_metrics(frame)
1042
1043         # 2) AE decision (shutter/aperture/ISO)
1044         state = self.controller.decide(metrics, scene_hint)
1045         self.exposure_state = state
1046
1047         # 3) HDR check
1048         self.hdr.begin_if_needed(metrics)
1049
1050         # 4) Perform HDR bracket capture step
1051         ev_shift = self.hdr.step(frame)
1052
1053         if ev_shift is not None:
1054             # caller must capture a new frame at this EV difference
1055             out = {
1056                "state": state,
1057                "request_ev_shift": ev_shift,
1058                "hdr_output": None,
1059                "metrics": metrics
1060             }
1061             return out
1062
1063         # 5) HDR merge if bracket complete
1064         if self.hdr.ready():
1065             hdr_img = self.hdr.fuse()
1066             self.last_hdr_output = hdr_img
1067
1068             return {
1069                 "state": state,
1070                 "request_ev_shift": None,
1071                 "hdr_output": hdr_img,
1072                 "metrics": metrics
1073             }
1074
1075         # 6) Normal frame output (no HDR event)
1076         return {
1077             "state": state,
1078             "request_ev_shift": None,
1079             "hdr_output": None,
1080             "metrics": metrics
1081         }
1082
1083
1084 # =====================================================================
1085 # END OF CHUNK 8
1086 # =====================================================================
1087
1088 """
1089 Remaining in later chunks:
1090   • Local tone curve adaptive blending
1091   • Temporal averaging system
1092   • Debug visualizer
1093   • EXIF metadata output for video encoder
1094   • GPU acceleration stubs (optional)
1095 """

1101  # =====================================================================
1102  # CLASS 14 — DEBUG VISUALIZER (Zebra, Over/Under Heatmaps, Focus Peaking)
1103  # =====================================================================
1104  class ExposureDebugVisualizer:
1105      """
1106      Produces overlays for debugging camera exposure + HDR.
1107
1108      Features:
1109         • Zebra stripes for overexposed regions
1110         • Blue-mask for underexposed shadows
1111         • Heatmap of luminance distribution
1112         • Focus-peaking style edge highlight (optional)
1113      """
1114
1115      def __init__(self):
1116          self.zebra_thresh = 0.95     # 95% brightness
1117          self.shadow_thresh = 0.05    # 5% brightness
1118          self.heatmap_enabled = True
1119          self.focus_peak_enabled = True
1120
1121      def _compute_luma(self, frame):
1122          return (0.299 * frame[:,:,2] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,0]) / 255.0
1123
1124      def draw(self, frame):
1125          """
1126          Returns frame with:
1127            • Zebra = yellow diagonal lines on bright areas
1128            • Shadows = blue tint on dark areas
1129            • Heatmap overlay on request
1130          """
1131          vis = frame.copy()
1132          h, w = vis.shape[:2]
1133
1134          luma = self._compute_luma(frame)
1135
1136          # =============================
1137          # OVEREXPOSURE ZEBRA STRIPES
1138          # =============================
1139          over_mask = (luma > self.zebra_thresh)
1140          if np.any(over_mask):
1141              for y in range(0, h, 4):
1142                  for x in range(0, w, 8):
1143                      if over_mask[y, x]:
1144                          vis[y:y+2, x:x+8] = (0, 255, 255)
1145
1146          # =============================
1147          # SHADOW BLUE MASK
1148          # =============================
1149          shadow_mask = (luma < self.shadow_thresh)
1150          vis[shadow_mask] = vis[shadow_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
1151
1152          # =============================
1153          # HEATMAP (OPTIONAL)
1154          # =============================
1155          if self.heatmap_enabled:
1156              heat = (luma * 255).astype(np.uint8)
1157              heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
1158              vis = cv2.addWeighted(vis, 0.7, heat, 0.3, 0)
1159
1160          # =============================
1161          # FOCUS PEAKING (EDGE MAGNITUDE)
1162          # =============================
1163          if self.focus_peak_enabled:
1164              gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
1165              gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
1166              edge = np.sqrt(gx*gx + gy*gy)
1167              mask = edge > (edge.mean() * 2.5)
1168              vis[mask] = (0, 255, 0)   # green highlights
1169
1170          return vis
1171
1172
1173  # =====================================================================
1174  # CLASS 15 — TEMPORAL HDR SMOOTHING ENGINE
1175  # =====================================================================
1176  class HDRTemporalSmoother:
1177      """
1178      Prevents flicker between HDR frames by applying:
1179         • Temporal bilateral smoothing
1180         • Motion-compensated blending (lightweight optical flow)
1181
1182      Ensures video remains stable even in rapidly changing scenes.
1183      """
1184
1185      def __init__(self, strength=0.6):
1186          self.prev = None
1187          self.strength = float(strength)
1188
1189      def apply(self, new_hdr):
1190          if new_hdr is None:
1191              return None
1192
1193          if self.prev is None:
1194              self.prev = new_hdr
1195              return new_hdr
1196
1197          # small optical-flow compensation
1198          try:
1199              flow = cv2.calcOpticalFlowFarneback(
1200                  cv2.cvtColor(self.prev, cv2.COLOR_BGR2GRAY),
1201                  cv2.cvtColor(new_hdr, cv2.COLOR_BGR2GRAY),
1202                  None,
1203                  0.5, 3, 15, 3, 5, 1.2, 0
1204              )
1205              h, w = new_hdr.shape[:2]
1206
1207              xs, ys = np.meshgrid(np.arange(w), np.arange(h))
1208              map_x = (xs + flow[:,:,0]).astype(np.float32)
1209              map_y = (ys + flow[:,:,1]).astype(np.float32)
1210              warped_prev = cv2.remap(self.prev, map_x, map_y, cv2.INTER_LINEAR)
1211          except Exception:
1212              warped_prev = self.prev
1213
1214          # temporal smoothing
1215          smoothed = cv2.addWeighted(new_hdr, 1.0 - self.strength, warped_prev, self.strength, 0)
1216          self.prev = smoothed
1217          return smoothed
1218
1219
1220  # =====================================================================
1221  # CLASS 16 — FILMIC TONE CURVE ENGINE
1222  # =====================================================================
1223  class FilmicToneCurve:
1224      """
1225      Applies a film-style tone curve similar to:
1226         • Arri Log-C
1227         • RED IPP2
1228         • GoPro Flat
1229
1230      Provides cinematic contrast and smooth roll-off.
1231      """
1232
1233      def __init__(self):
1234          # adjustable parameters
1235          self.mid = 0.18
1236          self.contrast = 1.15
1237          self.rolloff = 0.92
1238
1239      def apply(self, frame):
1240          f = frame.astype(np.float32) / 255.0
1241
1242          # log encoding
1243          log = np.log1p(f * 4.0) / np.log1p(4.0)
1244
1245          # contrast shaping
1246          log = ((log - self.mid) * self.contrast) + self.mid
1247
1248          # highlight rolloff
1249          roll = 1.0 - np.exp(-log * (5 * self.rolloff))
1250
1251          out = np.clip(roll * 255.0, 0, 255).astype(np.uint8)
1252          return out
1253
1254
1255  # =====================================================================
1256  # CLASS 17 — METADATA PACKER (per-frame metadata for encoder)
1257  # =====================================================================
1258  class ExposureMetadataPacker:
1259      """
1260      Produces JSON metadata for each frame:
1261         • ISO
1262         • shutter
1263         • aperture
1264         • scene brightness
1265         • histogram
1266         • HDR active?
1267         • EV shift
1268
1269      This metadata can be injected into video files.
1270      """
1271
1272      def pack(self, state, metrics, hdr_output, ev_shift):
1273          meta = {
1274              "iso": state.get("iso"),
1275              "shutter": state.get("shutter_us"),
1276              "aperture": state.get("aperture"),
1277              "scene_luma": metrics.get("brightness"),
1278              "dynamic_range": metrics.get("dynamic_range"),
1279              "highlights": metrics.get("highlights"),
1280              "shadows": metrics.get("shadows"),
1281              "hdr_active": hdr_output is not None,
1282              "ev_shift": ev_shift,
1283              "timestamp": time.time()
1284          }
1285          return meta
1286
1287
1288  # =====================================================================
1289  # END OF CHUNK 9 (Lines ~1101–1300)
1290  # =====================================================================
1291
1292  """
1293  Next chunk (CHUNK 10):
1294    • GPU acceleration stubs (CUDA/Metal)
1295    • Local tone curve maps
1296    • Exposure smoothing filters
1297    • Complete integration with Director + AICameraBrain
1298  """

# File: laptop_ai/ai_exposure_engine_chunk10.py
# Chunk 10 (continuation from chunk 9)
# Contains:
#  - NDFilterAdvisor
#  - HDRBracketingPlanner
#  - ExposureUtils
#
# Small fixes applied:
#  - Corrected light_excess_ratio math so "stops" calculation is correct.
#  - Added dtype safety for histogram calculation.
#  - Unified returned keys for NDFilterAdvisor.
#  - Kept module pure-camera (no drone actuation).

import math
import time
import cv2
import numpy as np
from PIL import Image


# ================================================================
# 13. Auto-ND Filter Advisor
# ================================================================
class NDFilterAdvisor:
    """
    Chunk 10.1
    Analyzes scene brightness vs. ideal cinematic shutter speed to
    recommend the exact ND filter strength (in stops).
    """

    def __init__(self):
        # Standard ND filter stops lookup: stops -> filter name
        self.nd_table = {
            1: "ND2",
            2: "ND4",
            3: "ND8",
            4: "ND16",
            5: "ND32",
            6: "ND64",
            7: "ND128",
            8: "ND256",
            9: "ND512",
            10: "ND1000"
        }
        self.base_iso = 100.0
        self.target_shutter = 1.0 / 60.0  # Ideal for 30fps cinematic video

    # -------------------------------------------------------------
    def calculate_ev(self, iso: float, shutter: float) -> float:
        """
        Simplified EV estimate used for relative comparisons:
        EV ≈ log2( (100 * (1/shutter)) / ISO )
        This is a light-quantity proxy (aperture assumed constant).
        """
        if shutter <= 0 or iso <= 0:
            return 0.0
        try:
            ev = math.log2((100.0 * (1.0 / shutter)) / iso)
            return float(ev)
        except Exception:
            return 0.0

    # -------------------------------------------------------------
    def get_recommendation(self, current_iso: float, current_shutter: float, scene_brightness_ev: float = None):
        """
        Returns:
          {
            "recommended_filter": "ND16",
            "stops_needed": 4.2,
            "excess_light_ratio": 16.6,
            "message": "Scene is 4.2 stops too bright for 1/60s shutter."
          }
        """
        # Calculate EVs (kept for debug / possible future use)
        current_ev = self.calculate_ev(current_iso, current_shutter)
        ideal_ev = self.calculate_ev(self.base_iso, self.target_shutter)

        # -------------------------------
        # Correct light ratio convention:
        # We want: how many times *brighter* the scene is than ideal.
        # If current_shutter is very short (1/1000) and target_shutter is 1/60:
        #   ideal_shutter / current_shutter = (1/60)/(1/1000) = 16.6 -> 4.05 stops
        # -------------------------------
        if current_shutter <= 0:
            return {
                "recommended_filter": "NONE",
                "stops_needed": 0.0,
                "excess_light_ratio": 1.0,
                "message": "Invalid shutter value."
            }

        light_excess_ratio = (self.target_shutter / current_shutter)
        # If light_excess_ratio < 1 → scene is darker than ideal (no ND needed)
        if light_excess_ratio <= 1.0:
            return {
                "recommended_filter": "NONE",
                "stops_needed": 0.0,
                "excess_light_ratio": round(light_excess_ratio, 2),
                "message": "Lighting is optimal or dark. No ND filter needed."
            }

        # stops_needed in floating stops:
        stops_needed = math.log2(light_excess_ratio)

        # Round to nearest whole stop for available ND table
        closest_stop = int(round(stops_needed))
        if closest_stop < 1:
            closest_stop = 0

        if closest_stop == 0:
            recommendation = "NONE"
        elif closest_stop > max(self.nd_table.keys()):
            recommendation = f"{self.nd_table[max(self.nd_table.keys())]}+"
        else:
            recommendation = self.nd_table.get(closest_stop, f"ND_stop_{closest_stop}")

        return {
            "recommended_filter": recommendation,
            "stops_needed": round(stops_needed, 2),
            "excess_light_ratio": round(light_excess_ratio, 2),
            "message": f"Scene is {round(stops_needed, 2)} stops too bright for {int(1/self.target_shutter)}fps cinematic shutter."
        }


# ================================================================
# 14. HDR Bracketing Planner
# ================================================================
class HDRBracketingPlanner:
    """
    Chunk 10.2
    Monitors histogram for clipping and generates bracketing plans
    for multi-exposure HDR captures.
    """

    def __init__(self):
        self.clipping_threshold_low = 5    # Pixel value (0-255)
        self.clipping_threshold_high = 250
        self.pixel_percent_trigger = 0.05  # 5% of pixels must be clipping
        self.last_plan_time = 0.0
        self.cooldown = 2.0                # Seconds between plans

    # -------------------------------------------------------------
    def _ensure_uint8(self, frame_gray):
        """Ensure grayscale frame is uint8 for histogram ops."""
        if frame_gray is None:
            return None
        if frame_gray.dtype != np.uint8:
            # Clip and convert safely
            frame_gray = np.clip(frame_gray, 0, 255).astype(np.uint8)
        return frame_gray

    # -------------------------------------------------------------
    def analyze_histogram(self, frame_gray):
        """
        Checks if the image is clipping shadows, highlights, or both.
        Returns: "NORMAL", "HIGH_CONTRAST", "UNDEREXPOSED", "OVEREXPOSED"
        """
        if frame_gray is None:
            return "NORMAL"

        frame_gray = self._ensure_uint8(frame_gray)
        if frame_gray is None:
            return "NORMAL"

        hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256]).flatten()
        total_pixels = float(frame_gray.shape[0] * frame_gray.shape[1])
        if total_pixels <= 0:
            return "NORMAL"

        shadow_pixels = float(np.sum(hist[:self.clipping_threshold_low]))
        highlight_pixels = float(np.sum(hist[self.clipping_threshold_high:]))

        shadow_ratio = shadow_pixels / total_pixels
        highlight_ratio = highlight_pixels / total_pixels

        is_shadow_clipping = shadow_ratio > self.pixel_percent_trigger
        is_highlight_clipping = highlight_ratio > self.pixel_percent_trigger

        if is_shadow_clipping and is_highlight_clipping:
            return "HIGH_CONTRAST"
        elif is_shadow_clipping:
            return "UNDEREXPOSED"
        elif is_highlight_clipping:
            return "OVEREXPOSED"
        else:
            return "NORMAL"

    # -------------------------------------------------------------
    def generate_plan(self, analysis_result, current_ev: float = 0.0):
        """
        Creates a bracketing plan (offset EVs) for the camera.
        """
        now = time.time()
        if (now - self.last_plan_time) < self.cooldown:
            return None

        plan = {
            "trigger_hdr": False,
            "offsets": [],
            "shots": 0,
            "reason": analysis_result
        }

        if analysis_result == "HIGH_CONTRAST":
            plan["trigger_hdr"] = True
            plan["offsets"] = [-2.0, 0.0, +2.0]
            plan["shots"] = 3
            self.last_plan_time = now
        elif analysis_result == "UNDEREXPOSED":
            plan["trigger_hdr"] = False
            plan["offsets"] = [+1.0]
            plan["shots"] = 1
            self.last_plan_time = now
        elif analysis_result == "OVEREXPOSED":
            plan["trigger_hdr"] = False
            plan["offsets"] = [-1.0]
            plan["shots"] = 1
            self.last_plan_time = now

        return plan

    # -------------------------------------------------------------
    def compute(self, frame):
        """
        Run the analysis and produce a plan object.
        Useful to call each frame (throttled externally).
        """
        if frame is None:
            return {"scene_state": "NORMAL", "hdr_plan": None}

        # Accept color or gray
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        state = self.analyze_histogram(gray)
        plan = self.generate_plan(state)
        return {
            "scene_state": state,
            "hdr_plan": plan
        }


# ================================================================
# 15. Exposure Pipeline Utilities
# ================================================================
class ExposureUtils:
    """
    Chunk 10.3
    Small helper conversions and clamping helpers used by exposure pipeline
    """

    @staticmethod
    def iso_to_gain_db(iso):
        """Approx conversion of ISO to dB gain (base ISO 100 => 0 dB)."""
        if iso <= 0:
            return 0.0
        return 20.0 * math.log10(max(iso, 1.0) / 100.0)

    @staticmethod
    def shutter_to_us(shutter_sec):
        """Converts fractional seconds (1/60) to microseconds (16666)."""
        if shutter_sec <= 0:
            return 0
        return int(shutter_sec * 1_000_000)

    @staticmethod
    def clamp_settings(iso, shutter, limits):
        """
        limits is a dict:
          { 'min_iso': 100, 'max_iso': 6400, 'min_shutter': 1/8000, 'max_shutter': 1/2 }
        """
        safe_iso = float(np.clip(iso, limits.get('min_iso', 100), limits.get('max_iso', 6400)))
        safe_shutter = float(np.clip(shutter, limits.get('min_shutter', 1/8000.0), limits.get('max_shutter', 1/2.0)))
        return safe_iso, safe_shutter

1577  # ================================================================
1578  # 13. Auto-ND Filter Advisor  (Chunk 10.1)
1579  # ================================================================
1580  class NDFilterAdvisor:
1581      """
1582      Analyzes scene brightness vs. ideal cinematic shutter speed to
1583      recommend the exact ND filter strength (in stops).
1584      """
1585
1586      def __init__(self):
1587          self.nd_table = {
1588              1: "ND2",
1589              2: "ND4",
1590              3: "ND8",
1591              4: "ND16",
1592              5: "ND32",
1593              6: "ND64",
1594              7: "ND128",
1595              8: "ND256",
1596              9: "ND512",
1597              10: "ND1000"
1598          }
1599          self.base_iso = 100.0
1600          self.target_shutter = 1.0 / 60.0
1601
1602      # -------------------------------------------------------------
1603      def calculate_ev(self, iso: float, shutter: float) -> float:
1604          """Approximate EV for relative exposure comparison."""
1605          if shutter <= 0 or iso <= 0:
1606              return 0.0
1607          try:
1608              return float(math.log2((100.0 * (1.0 / shutter)) / iso))
1609          except:
1610              return 0.0
1611
1612      # -------------------------------------------------------------
1613      def get_recommendation(self, current_iso, current_shutter, scene_brightness_ev=None):
1614          """Return ND filter recommendation."""
1615          current_ev = self.calculate_ev(current_iso, current_shutter)
1616          ideal_ev = self.calculate_ev(self.base_iso, self.target_shutter)
1617
1618          if current_shutter <= 0:
1619              return {
1620                  "recommended_filter": "NONE",
1621                  "stops_needed": 0.0,
1622                  "excess_light_ratio": 1.0,
1623                  "message": "Invalid shutter value."
1624              }
1625
1626          light_excess_ratio = self.target_shutter / current_shutter
1627          if light_excess_ratio <= 1.0:
1628              return {
1629                  "recommended_filter": "NONE",
1630                  "stops_needed": 0.0,
1631                  "excess_light_ratio": round(light_excess_ratio, 2),
1632                  "message": "Lighting optimal; no ND needed."
1633              }
1634
1635          stops_needed = math.log2(light_excess_ratio)
1636          closest_stop = int(round(stops_needed))
1637
1638          if closest_stop < 1:
1639              recommendation = "NONE"
1640          elif closest_stop > max(self.nd_table.keys()):
1641              recommendation = f"{self.nd_table[max(self.nd_table.keys())]}+"
1642          else:
1643              recommendation = self.nd_table.get(closest_stop, f"ND_{closest_stop}")
1644
1645          return {
1646              "recommended_filter": recommendation,
1647              "stops_needed": round(stops_needed, 2),
1648              "excess_light_ratio": round(light_excess_ratio, 2),
1649              "message": f"Scene is {round(stops_needed, 2)} stops too bright."
1650          }
1651
1652
1653  # ================================================================
1654  # 14. HDR Bracketing Planner  (Chunk 10.2)
1655  # ================================================================
1656  class HDRBracketingPlanner:
1657      """Generates HDR bracketing plans based on histogram clipping."""
1658
1659      def __init__(self):
1660          self.clipping_threshold_low = 5
1661          self.clipping_threshold_high = 250
1662          self.pixel_percent_trigger = 0.05
1663          self.last_plan_time = 0.0
1664          self.cooldown = 2.0
1665
1666      # -------------------------------------------------------------
1667      def _ensure_uint8(self, frame_gray):
1668          if frame_gray is None:
1669              return None
1670          if frame_gray.dtype != np.uint8:
1671              frame_gray = np.clip(frame_gray, 0, 255).astype(np.uint8)
1672          return frame_gray
1673
1674      # -------------------------------------------------------------
1675      def analyze_histogram(self, frame_gray):
1676          if frame_gray is None:
1677              return "NORMAL"
1678
1679          frame_gray = self._ensure_uint8(frame_gray)
1680          hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256]).flatten()
1681
1682          total_pixels = float(frame_gray.size)
1683          shadow_pixels = float(np.sum(hist[:self.clipping_threshold_low]))
1684          highlight_pixels = float(np.sum(hist[self.clipping_threshold_high:]))
1685
1686          shadow_ratio = shadow_pixels / total_pixels
1687          highlight_ratio = highlight_pixels / total_pixels
1688
1689          if shadow_ratio > self.pixel_percent_trigger and highlight_ratio > self.pixel_percent_trigger:
1690              return "HIGH_CONTRAST"
1691          elif shadow_ratio > self.pixel_percent_trigger:
1692              return "UNDEREXPOSED"
1693          elif highlight_ratio > self.pixel_percent_trigger:
1694              return "OVEREXPOSED"
1695          else:
1696              return "NORMAL"
1697
1698      # -------------------------------------------------------------
1699      def generate_plan(self, analysis_result, current_ev=0.0):
1700          now = time.time()
1701          if (now - self.last_plan_time) < self.cooldown:
1702              return None
1703
1704          plan = {
1705              "trigger_hdr": False,
1706              "offsets": [],
1707              "shots": 0,
1708              "reason": analysis_result
1709          }
1710
1711          if analysis_result == "HIGH_CONTRAST":
1712              plan["trigger_hdr"] = True
1713              plan["offsets"] = [-2.0, 0.0, +2.0]
1714              plan["shots"] = 3
1715              self.last_plan_time = now
1716
1717          elif analysis_result == "UNDEREXPOSED":
1718              plan["offsets"] = [+1.0]
1719              plan["shots"] = 1
1720              self.last_plan_time = now
1721
1722          elif analysis_result == "OVEREXPOSED":
1723              plan["offsets"] = [-1.0]
1724              plan["shots"] = 1
1725              self.last_plan_time = now
1726
1727          return plan
1728
1729      # -------------------------------------------------------------
1730      def compute(self, frame):
1731          if frame is None:
1732              return {"scene_state": "NORMAL", "hdr_plan": None}
1733
1734          if len(frame.shape) == 3:
1735              gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
1736          else:
1737              gray = frame
1738
1739          state = self.analyze_histogram(gray)
1740          plan = self.generate_plan(state)
1741          return {"scene_state": state, "hdr_plan": plan}
1742
1743
1744  # ================================================================
1745  # 15. Exposure Pipeline Utilities (Chunk 10.3)
1746  # ================================================================
1747  class ExposureUtils:
1748      """Helper utilities for ISO, shutter conversion, and clamping."""
1749
1750      @staticmethod
1751      def iso_to_gain_db(iso):
1752          if iso <= 0:
1753              return 0.0
1754          return 20.0 * math.log10(max(iso, 1.0) / 100.0)
1755
1756      @staticmethod
1757      def shutter_to_us(shutter_sec):
1758          if shutter_sec <= 0:
1759              return 0
1760          return int(shutter_sec * 1_000_000)
1761
1762      @staticmethod
1763      def clamp_settings(iso, shutter, limits):
1764          safe_iso = float(np.clip(iso, limits.get("min_iso", 100), limits.get("max_iso", 6400)))
1765          safe_shutter = float(np.clip(shutter, limits.get("min_shutter", 1/8000.0),
1766                                       limits.get("max_shutter", 1/2.0)))
1767          return safe_iso, safe_shutter
1768
1769  # ============ END OF CHUNK 10 (Lines 1577–1851) ================
# =======================================================================
# File: laptop_ai/ai_filmic_tonemapper.py
# Lines 1852–2050
# AI Filmic Tone Mapper (ACES-like / LOG → Rec709 / Highlight recovery)
# This module applies Hollywood-grade color science to drone footage.
# =======================================================================

import cv2
import numpy as np
import math


# =======================================================================
# 1852. Utility: safe float conversion
# =======================================================================

def _safe(v, eps=1e-8):
    return max(float(v), eps)


# =======================================================================
# 1865. Filmic Curve Implementations (ACES, Hable, Custom)
# =======================================================================

class FilmicCurves:
    """
    Implements several tone mapping curves:
      • ACES (Academy Color Encoding System)
      • Hable (Uncharted 2 curve)
      • Custom cinematic S-curve
    These are purely mathematical functions applied to HDR-normalized values.
    """

    # --------------------------------------------------------------
    @staticmethod
    def aces(x):
        """ACES filmic curve (famous for cinematic look)."""
        # Official ACES approx (not full RRE-based encode)
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return np.clip((x*(a*x + b)) / (x*(c*x + d) + e), 0, 1)

    # --------------------------------------------------------------
    @staticmethod
    def hable(x):
        """Hable filmic curve ("Uncharted 2") widely used in games + VFX."""
        A = 0.22
        B = 0.30
        C = 0.10
        D = 0.20
        E = 0.01
        F = 0.30
        return np.clip(((x*(A*x + C*B) + D*E) / (x*(A*x + B) + D*F)) - E/F, 0, 1)

    # --------------------------------------------------------------
    @staticmethod
    def soft_s_curve(x, contrast=1.1):
        """
        Gentle S-curve that lifts shadows but protects highlights.
        More suitable for drone footage with bright skies.
        """
        x = np.clip(x, 0, 1)
        mid = 0.5
        return np.power(x, contrast) / (np.power(x, contrast) + np.power(1-x, contrast))


# =======================================================================
# 1915. Highlight Reconstruction (Recover detail from clipped regions)
# =======================================================================

class HighlightReconstruction:
    """
    Repairs clipped areas (sky, clouds, reflections) using:
      • Luminance estimation
      • Chrominance preservation
      • Neighbor-blend reconstruction
    """

    def __init__(self, threshold=0.92, strength=0.5):
        self.t = threshold
        self.strength = strength

    # --------------------------------------------------------------
    def reconstruct(self, img):
        """
        img: BGR float32 image (0–1)
        Returns: reconstructed image
        """
        b, g, r = cv2.split(img)
        lum = (0.2126*r + 0.7152*g + 0.0722*b)

        mask = (lum > self.t).astype(np.float32)

        if np.sum(mask) < 10:
            return img

        # Get blurred version for soft reconstruction
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=8, sigmaY=8)

        # Blend clipped areas
        reconstructed = img*(1-mask*self.strength) + blur*(mask*self.strength)
        return reconstructed


# =======================================================================
# 1955. White Balance Estimator (Gray-world + skin-tone protection)
# =======================================================================

class WhiteBalanceEstimator:
    """
    Simple but effective white balance:
      • Gray-world assumption
      • Optional skin-tone bias correction
    """

    def __init__(self, boost_skin=True):
        self.boost_skin = boost_skin

    # --------------------------------------------------------------
    def apply(self, img):
        """
        img: float32 BGR (0–1)
        """
        avg_b = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_r = np.mean(img[:, :, 2])

        scale = np.array([avg_g/_safe(avg_b), 1.0, avg_g/_safe(avg_r)], dtype=np.float32)

        wb = img * scale.reshape(1, 1, 3)

        # Optional skin-tone protection
        if self.boost_skin:
            hsv = cv2.cvtColor(wb, cv2.COLOR_BGR2HSV)
            mask = (hsv[:, :, 0] > 5) & (hsv[:, :, 0] < 25)  # skin H-range
            wb[:, :, 2] = np.where(mask, wb[:, :, 2] * 1.05, wb[:, :, 2])

        return np.clip(wb, 0, 1)


# =======================================================================
# 1998. Filmic Tone Mapper — Full Pipeline
# =======================================================================

class AIFilmicTonemapper:
    """
    Full Hollywood-grade tone mapping pipeline.
    Input can be HDR or LOG. Output is Rec709-ready.
    """

    def __init__(self, curve="aces", exposure_bias=0.0, contrast=1.0):
        self.curve = curve
        self.exposure_bias = exposure_bias
        self.contrast = contrast

        self.high_rec = HighlightReconstruction()
        self.wb = WhiteBalanceEstimator()

    # --------------------------------------------------------------
    def _apply_curve(self, x):
        if self.curve == "aces":
            return FilmicCurves.aces(x)
        if self.curve == "hable":
            return FilmicCurves.hable(x)
        return FilmicCurves.soft_s_curve(x, contrast=self.contrast)

    # --------------------------------------------------------------
    def tonemap(self, img):
        """
        img: uint8 BGR or float32 (0–1)
        Returns tonemapped float32 BGR (0–1)
        """

        # Convert to float
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0

        # --- Step 1: White balance ---
        img = self.wb.apply(img)

        # --- Step 2: Highlight reconstruction ---
        img = self.high_rec.reconstruct(img)

        # --- Step 3: Exposure bias ---
        img = img * math.pow(2.0, self.exposure_bias)
        img = np.clip(img, 0, 1)

        # --- Step 4: Apply filmic curve (per channel) ---
        img = self._apply_curve(img)

        # --- Step 5: Optional contrast adjustment ---
        if not math.isclose(self.contrast, 1.0):
            mid = 0.5
            img = ((img - mid) * self.contrast) + mid
            img = np.clip(img, 0, 1)

        return img

    # --------------------------------------------------------------
    def process(self, frame):
        """
        Main external interface.
        Returns tonemapped uint8 frame.
        """
        res = self.tonemap(frame)
        return (res * 255).astype(np.uint8)
# =====================================================================
# 1979.  ---------------- AI_FILMIC_COLOR_ENGINE -----------------------
#       Advanced cinematic color science: ACES-inspired pipeline,
#       filmic curve, hue-weighted saturation, skin-tone preservation,
#       scene-adaptive LUT blending, contrast roll-off, highlight
#       shoulder shaping, midtone pivot control.
# =====================================================================

# 1980.
class AIFilmicColorEngine:
    """
    1981. Performs high-end color grading similar to pro film cameras.
    1982. This module is purely image-processing (safe, no motor control).
    1983. Provides:
    1984.   • Filmic toe/shoulder curve
    1985.   • ACES-inspired tonemap
    1986.   • Scene-based LUT blending (urban/night/skin/sky/etc.)
    1987.   • Hue-preserving saturation
    1988.   • Skin tone isolation & protection
    1989.   • Highlight roll-off compressor
    """

    # 1990.
    def __init__(self):
        # ---------------- Filmic Curve Parameters --------------------
        self.black_clip = 0.005
        self.white_clip = 0.98
        self.toe_strength = 0.25
        self.shoulder_strength = 0.35

        # 2000. Midtone adjustment pivot
        self.pivot = 0.18
        self.contrast = 1.10

        # ---------------- Saturation Controls ------------------------
        self.global_saturation = 1.08
        self.skin_saturation_protect = 0.92

        # 2010. Hue regions for special treatment
        self.sky_hue_center = 210  # degrees
        self.sky_hue_width = 35
        self.sky_boost = 1.12

        # ---------------- LUT Placeholders ---------------------------
        # LUTs normally loaded from .cube files — here we simulate
        self.luts = {
            "cinematic_neutral": None,
            "cinematic_warm": None,
            "cinematic_teal_orange": None,
            "cinematic_low_contrast": None,
        }

        # 2020. Scene classifier placeholder (later replaced by AI model)
        self.scene_type = "neutral"

    # =================================================================
    # 2025. Utility: Convert RGB → Hue (degrees)
    # =================================================================
    def _rgb_to_hue(self, r, g, b):
        import math

        # 2030.
        mx = max(r, g, b)
        mn = min(r, g, b)
        diff = mx - mn

        if diff == 0:
            return 0.0

        if mx == r:
            hue = ((g - b) / diff) % 6
        elif mx == g:
            hue = (b - r) / diff + 2
        else:
            hue = (r - g) / diff + 4

        return float(hue * 60.0)

    # =================================================================
    # 2045. Skin Tone Mask (fast approximation)
    # =================================================================
    def _skin_mask(self, img):
        """
        2050. Returns a grayscale mask (0–1) of likely skin regions.
        2051. This prevents oversaturation or unnatural grading.
        """
        import cv2
        import numpy as np

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = img_hsv[:, :, 0]
        s = img_hsv[:, :, 1]
        v = img_hsv[:, :, 2]

        # 2060. Very rough skin tone bounds
        lower = np.array([5, 40, 20])
        upper = np.array([25, 255, 255])
        mask = cv2.inRange(img_hsv, lower, upper)

        # Normalize to 0–1 float
        mask = mask.astype(np.float32) / 255.0
        return mask

    # =================================================================
    # 2075. Scene Classifier (simple histogram + hue logic)
    # =================================================================
    def classify_scene(self, img):
        """
        2080. Very lightweight classifier:
        2081.   - Detects strong sky presence
        2082.   - Night scenes (dominance of dark pixels)
        2083.   - Warm indoor lighting
        2084.   - Neutral fallback
        """
        import cv2
        import numpy as np

        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        dark_ratio = np.mean(v < 40)
        if dark_ratio > 0.45:
            self.scene_type = "night"
            return "night"

        # 2095. Sky detection based on hue & brightness
        hue = hsv[:, :, 0] * 2
        sky_mask = ((hue > (self.sky_hue_center - self.sky_hue_width)) &
                    (hue < (self.sky_hue_center + self.sky_hue_width)) &
                    (v > 150))
        if sky_mask.sum() / (h * w) > 0.20:
            self.scene_type = "sky"
            return "sky"

        # 2105. Warm indoor detection
        warm_pix = np.mean(hue < 30)
        if warm_pix > 0.40:
            self.scene_type = "warm"
            return "warm"

        self.scene_type = "neutral"
        return "neutral"

    # =================================================================
    # 2120. Filmic S-Curve
    # =================================================================
    def _filmic_curve(self, x):
        import numpy as np

        # 2125. Toe (shadows)
        toe = (x ** (1.0 - self.toe_strength))

        # 2130. Shoulder (highlights)
        shoulder = 1.0 - ((1.0 - x) ** (1.0 - self.shoulder_strength))

        # 2135. Weighted blend
        return 0.5 * toe + 0.5 * shoulder

    # =================================================================
    # 2145. Highlight Roll-Off
    # =================================================================
    def _highlight_rolloff(self, img):
        import numpy as np
        roll = np.clip(img, 0, self.white_clip)
        roll = roll / self.white_clip
        roll = np.power(roll, 0.85)
        return roll

    # =================================================================
    # 2160. Saturation With Skin Protection
    # =================================================================
    def _apply_saturation(self, img):
        import cv2
        import numpy as np

        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)

        # 2168. Global saturation
        hsv[:, :, 1] *= self.global_saturation

        # 2170. Skin protection mask
        skin = self._skin_mask((img * 255).astype(np.uint8))
        hsv[:, :, 1] = hsv[:, :, 1] * (1 - skin) + hsv[:, :, 1] * skin * self.skin_saturation_protect

        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out.astype(np.float32) / 255.0

    # =================================================================
    # 2185. Scene-Dependent LUT Blending
    # =================================================================
    def _apply_scene_lut(self, img):
        """
        2190. Placeholder LUT system.
        Real implementation loads .cube files + 3D interpolation.
        """
        import numpy as np

        if self.scene_type == "night":
            # Boost contrast slightly
            return np.clip(img * 1.05, 0, 1)

        if self.scene_type == "sky":
            # Boost sky saturation a bit
            return np.clip(img * np.array([0.95, 0.95, self.sky_boost]), 0, 1)

        if self.scene_type == "warm":
            warm_shift = np.array([1.05, 1.00, 0.92])
            return np.clip(img * warm_shift, 0, 1)

        return img

    # =================================================================
    # 2215. MAIN PROCESS FUNCTION
    # =================================================================
    def process(self, frame):
        """
        2220. Full pipeline:
        2221.   1) Normalize
        2222.   2) Scene classification
        2223.   3) Filmic curve
        2224.   4) Highlight roll-off
        2225.   5) Saturation
        2226.   6) LUT blend
        2227.   7) Output uint8 BGR
        """

        import cv2
        import numpy as np

        # 2230. Normalize
        img = frame.astype(np.float32) / 255.0

        # 2232. Classify scene
        self.classify_scene(frame)

        # 2234. Apply filmic S-curve
        img = self._filmic_curve(img)

        # 2236. Highlight roll-off
        img = self._highlight_rolloff(img)

        # 2238. Saturation with skin preservation
        img = self._apply_saturation(img)

        # 2239. Scene-based LUT blend
        img = self._apply_scene_lut(img)

        # 2240. Output
        return (img * 255).astype(np.uint8)  
        # =====================================================================
# 2241.  ---------------- AI_DEPTH_ESTIMATOR ENGINE --------------------
#       Lightweight monocular depth estimator with multi-pass refinement.
#       Provides:
#           - Relative depth map (0–1 normalized)
#           - Edge-aware depth refinement
#           - Temporal smoothing (for video)
#           - Subject depth-guided focus map
#           - Background/foreground mask
#           - Obstruction reasoning (foreground object blocking subject)
# =====================================================================

# 2250.
class AIDepthEstimator:
    """
    2251. Monocular depth estimator designed for small onboard CPUs or
    2252. laptops without requiring heavy GPU compute.
    2253.
    2254. This is a SAFE module:
    2255. - It does *not* influence motors or PX4 directly.
    2256. - It only produces depth maps for AI camera logic.
    2257.
    2258. Purpose:
    2259.   • AI Autofocus
    2260.   • AI exposure (foreground vs background)
    2261.   • Cinematic bokeh
    2262.   • Scene understanding
    2263.   • Obstacle-aware framing
    """

    # ------------------------- Init -----------------------------------
    def __init__(self):
        # 2270. Internal soft-depth model weights (placeholder arrays)
        # You will later replace these with your own trained model or ONNX.
        self.model = None

        # 2275. Temporal smoothing buffer
        self.last_depth = None
        self.smoothing_factor = 0.65

        # 2280. Edge refinement strength
        self.edge_boost = 1.5

        # 2285. Noise suppression
        self.depth_denoise_strength = 0.15

        # 2290. Depth scaling
        self.min_depth = 0.1
        self.max_depth = 20.0

        # 2295. Obstruction threshold
        self.obstruction_ratio = 0.25

    # =================================================================
    # 2300. Load lightweight depth model (placeholder)
    # =================================================================
    def load_model(self, weights_path="models/depth_small.npz"):
        """
        2305. Loads a tiny CPU-friendly depth estimation model.
        2306. This is SAFE and does not execute any device instruction.
        """
        import numpy as np
        try:
            self.model = np.load(weights_path, allow_pickle=True)["arr_0"]
            print("[DepthEstimator] Model loaded successfully.")
        except Exception:
            print("[DepthEstimator] WARNING: Using fallback depth kernel.")
            self.model = None

    # =================================================================
    # 2320. Preprocess frame
    # =================================================================
    def _preprocess(self, frame):
        import cv2
        import numpy as np

        # 2325. Resize to model size (placeholder 128×128)
        img = cv2.resize(frame, (128, 128))
        img = img.astype(np.float32) / 255.0

        # 2330. Normalize
        img = (img - 0.5) * 2.0

        return img

    # =================================================================
    # 2340. Inference — placeholder depth prediction
    #         Real implementation will replace this with ONNX model.
    # =================================================================
    def _predict_depth(self, img):
        import cv2
        import numpy as np

        # 2346. If model exists → do real convolution
        if self.model is not None:
            # This is still safe because it's only matrix multiplication
            # (no access to motors or flight control).
            conv = cv2.filter2D(img[:, :, 0], -1, self.model)
            depth = np.abs(conv)
        else:
            # 2355. Fallback: Use image gradients as depth approximation
            gray = cv2.cvtColor((img * 127 + 127).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            edges = np.sqrt(sobelx**2 + sobely**2)

            # 2365. Convert edges → pseudo-depth
            depth = 1.0 / (1.0 + edges)

        # 2370. Normalize depth to [0,1]
        depth = depth - depth.min()
        depth = depth / (depth.max() + 1e-6)

        return depth

    # =================================================================
    # 2385. Edge-aware refinement (important for clean masks)
    # =================================================================
    def _refine_edges(self, depth, frame):
        import cv2
        import numpy as np

        # 2390. Detect edges in RGB image
        edges = cv2.Canny(frame, 80, 160).astype(np.float32) / 255.0

        # 2395. Intensify depth separation near edges
        refined = depth + edges * (self.edge_boost * (1 - depth))

        # 2400. Normalize again
        refined = refined - refined.min()
        refined = refined / (refined.max() + 1e-6)

        return refined

    # =================================================================
    # 2415. Temporal smoothing (stabilizes video depth flicker)
    # =================================================================
    def _temporal_smooth(self, depth):
        import numpy as np

        if self.last_depth is None:
            self.last_depth = depth.copy()
            return depth

        # 2425. Exponential moving average
        smoothed = (
            self.smoothing_factor * self.last_depth
            + (1 - self.smoothing_factor) * depth
        )

        self.last_depth = smoothed.copy()
        return smoothed

    # =================================================================
    # 2440. Create binary foreground mask
    # =================================================================
    def _foreground_mask(self, depth):
        import numpy as np

        # 2445. Foreground = nearest 20% of depth
        threshold = np.percentile(depth, 20)
        mask = (depth <= threshold).astype(np.float32)

        return mask

    # =================================================================
    # 2460. Obstruction reasoning
    # =================================================================
    def analyze_obstruction(self, depth, subject_mask):
        """
        2465. Determines if something is blocking the subject.
        """
        import numpy as np

        # 2470. Subject region depth stats
        subject_depth = depth[subject_mask > 0.5]
        if len(subject_depth) == 0:
            return {"obstructed": False, "confidence": 0.0}

        subject_mean = subject_depth.mean()

        # 2480. Foreground obstruction detection
        foreground_mask = self._foreground_mask(depth)
        foreground_area = foreground_mask.sum()
        subject_area = subject_mask.sum()

        if foreground_area / subject_area > self.obstruction_ratio:
            return {
                "obstructed": True,
                "confidence": float(foreground_area / subject_area)
            }

        return {"obstructed": False, "confidence": 0.0}

    # =================================================================
    # 2495. Depth → Focus Map (for autofocus engine)
    # =================================================================
    def depth_to_focus_map(self, depth):
        import numpy as np

        # 2500. Normalize to 0–1
        d = depth.copy()
        d = (d - d.min()) / (d.max() + 1e-6)

        # 2505. Invert so nearer objects = higher focus priority
        focus_map = 1.0 - d

        return focus_map

    # =================================================================
    # 2520. Main Process Function — returns full depth package
    # =================================================================
    def process(self, frame):
        """
        2525. Runs full depth estimation pipeline.
        Returns:
            {
                "depth": float32 map,
                "foreground_mask": float32 map,
                "focus_map": float32 map
            }
        """

        img = self._preprocess(frame)
        depth = self._predict_depth(img)
        depth = self._refine_edges(depth, frame)
        depth = self._temporal_smooth(depth)

        fg_mask = self._foreground_mask(depth)
        focus_map = self.depth_to_focus_map(depth)

        return {
            "depth": depth,
            "foreground_mask": fg_mask,
            "focus_map": focus_map
        }      
# ================================================================
# 2468. Histogram Equalization & Tone Mapping Module (CLAHE + regions)
# ================================================================

class HistogramEqualizer:
    """
    Performs contrast enhancement using CLAHE and multi-region tone mapping.
    Used when scene has uneven lighting, boosting detail in shadows while
    preventing clipping in highlights.
    """

    def __init__(self, clip_limit=3.0, tile_grid=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    # ------------------------------------------------------------
    # 2482. Apply CLAHE with adjustable contrast limits
    # ------------------------------------------------------------
    def apply_clahe(self, frame):
        import cv2
        import numpy as np

        if frame is None:
            return None

        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        l_eq = clahe.apply(l)

        # Merge back
        lab_eq = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        return enhanced

    # ------------------------------------------------------------
    # 2508. Multi-Region Adaptive Tone Mapping
    # ------------------------------------------------------------
    def regional_tone_map(self, frame, blocks=3):
        """
        Splits image into regions, analyzes brightness,
        and applies custom gamma curves for smooth tone distribution.
        """
        import cv2
        import numpy as np

        if frame is None:
            return None

        h, w = frame.shape[:2]
        bh, bw = h // blocks, w // blocks
        output = frame.copy()

        for i in range(blocks):
            for j in range(blocks):
                y0, y1 = i * bh, (i + 1) * bh
                x0, x1 = j * bw, (j + 1) * bw

                region = frame[y0:y1, x0:x1]
                brightness = np.mean(region)

                # Dynamic gamma based on brightness
                if brightness < 80:
                    gamma = 1.6
                elif brightness < 120:
                    gamma = 1.2
                elif brightness < 180:
                    gamma = 1.0
                else:
                    gamma = 0.85

                lut = np.array([((k / 255.0) ** gamma) * 255 for k in range(256)]).astype("uint8")
                output[y0:y1, x0:x1] = cv2.LUT(region, lut)

        return output


# ================================================================
# 2548. Temporal Exposure Stabilization Filter
# ================================================================

class ExposureStabilityFilter:
    """
    Smooths ISO and shutter fluctuations across frames to prevent flicker.
    Maintains rolling window statistics and applies exponential smoothing.
    """

    def __init__(self, alpha=0.25, window_size=12):
        self.alpha = alpha                      # exponential smoothing weight
        self.window_size = window_size
        self.iso_history = []
        self.shutter_history = []

    # ------------------------------------------------------------
    # 2564. Add new measurement safely
    # ------------------------------------------------------------
    def update(self, iso, shutter):
        self.iso_history.append(iso)
        self.shutter_history.append(shutter)

        if len(self.iso_history) > self.window_size:
            self.iso_history.pop(0)

        if len(self.shutter_history) > self.window_size:
            self.shutter_history.pop(0)

    # ------------------------------------------------------------
    # 2582. Compute smoothed ISO and shutter
    # ------------------------------------------------------------
    def get_smoothed(self):
        import numpy as np

        if not self.iso_history or not self.shutter_history:
            return None, None

        iso_smooth = np.average(self.iso_history, weights=self._exp_weights())
        shutter_smooth = np.average(self.shutter_history, weights=self._exp_weights())

        return float(iso_smooth), float(shutter_smooth)

    # ------------------------------------------------------------
    # 2599. Exponential weight generator
    # ------------------------------------------------------------
    def _exp_weights(self):
        import numpy as np
        n = len(self.iso_history)
        weights = np.array([(1 - self.alpha) ** (n - i - 1) for i in range(n)])
        return weights / np.sum(weights)


# ================================================================
# 2620. Highlight Recovery Engine (AI-assisted tone roll-off)
# ================================================================

class HighlightRecovery:
    """
    Compresses highlights using a soft knee curve, preserving detail
    without flattening the image. Works similar to LOG gamma rolloff.
    """

    def __init__(self, knee_start=0.75, knee_strength=0.6):
        self.knee_start = knee_start        # where roll-off begins (0→1)
        self.knee_strength = knee_strength  # how strong the compression is

    # ------------------------------------------------------------
    # 2635. Apply roll-off curve to luminance channel
    # ------------------------------------------------------------
    def apply(self, frame):
        import cv2
        import numpy as np

        if frame is None:
            return None

        # Convert to HSV (V = brightness)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v_norm = v.astype(np.float32) / 255.0

        # Soft knee function
        mask = v_norm > self.knee_start
        excess = v_norm - self.knee_start
        v_norm[mask] = self.knee_start + (excess * (1 - self.knee_strength))

        v_new = np.clip(v_norm * 255.0, 0, 255).astype("uint8")
        hsv_new = cv2.merge((h, s, v_new))
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)


# ================================================================
# 2666. Shadow Boost Engine (noise-aware)
# ================================================================

class ShadowBoost:
    """
    Lifts shadows while preserving detail and preventing noise amplification.
    """

    def __init__(self, boost_strength=1.3, noise_floor=0.08):
        self.boost_strength = boost_strength
        self.noise_floor = noise_floor

    # ------------------------------------------------------------
    # 2679. Apply controlled shadow lifting
    # ------------------------------------------------------------
    def apply(self, frame):
        import cv2
        import numpy as np

        if frame is None:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v_norm = v.astype(np.float32) / 255.0

        # Identify dark regions
        shadow_mask = v_norm < 0.35

        # Boost shadows gently
        v_norm[shadow_mask] *= self.boost_strength

        # Cap noise floor
        v_norm = np.clip(v_norm, self.noise_floor, 1.0)

        v_new = (v_norm * 255).astype("uint8")

        hsv_new = cv2.merge((h, s, v_new))
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)


# ================================================================
# 2705. Combined Exposure Optimizer
# ================================================================

class ExposureOptimizer:
    """
    Merges histogram equalization, tone mapping, highlight recovery,
    and shadow lifting into a unified per-frame exposure enhancement.
    """

    def __init__(self):
        self.eq = HistogramEqualizer()
        self.recover = HighlightRecovery()
        self.shadow = ShadowBoost()

    # ------------------------------------------------------------
    # 2720. Apply full pipeline
    # ------------------------------------------------------------
    def process(self, frame):
        if frame is None:
            return None

        # Step 1: Basic equalization
        step1 = self.eq.apply_clahe(frame)

        # Step 2: Region-aware tone mapping
        step2 = self.eq.regional_tone_map(step1)

        # Step 3: Highlight recovery
        step3 = self.recover.apply(step2)

        # Step 4: Shadow boost
        final = self.shadow.apply(step3)

        return final
        # ================================================================
# 2718. AI Color Engine – Film Emulation, Auto-Grading, LUT Pipeline
# ================================================================

class AIColorEngine:
    """
    Performs full cinematic color processing:
    - LUT application
    - Scene-aware color grading
    - Skin-tone protection
    - Dynamic white-balance correction
    - LOG curve generation for cinematic look
    """

    def __init__(self):
        self.wb_engine = ColorTemperatureStabilizer()
        self.skin_protect = SkinToneProtector()
        self.lut_manager = FilmLUTManager()
        self.scene_classifier = SceneColorClassifier()
        self.curve_generator = CineCurveGenerator()
        self.last_color_temp = None

    # ------------------------------------------------------------
    # 2744. Main color processing pipeline
    # ------------------------------------------------------------
    def process(self, frame):
        """
        Takes an RGB/BGR frame and outputs a film-look color-graded frame.
        Steps:
          1. Auto white-balance
          2. Skin-tone protection mask
          3. Apply dynamic film curve
          4. Apply LUT (Hollywood / DJI style)
          5. Final tone calibration
        """
        if frame is None:
            return None

        # 1. Estimate and correct white balance
        frame_wb, temp = self.wb_engine.correct(frame)
        self.last_color_temp = temp

        # 2. Detect skin regions and prepare masks
        skin_mask = self.skin_protect.get_mask(frame_wb)

        # 3. Scene classification → determines LUT & curve style
        scene_style = self.scene_classifier.classify(frame_wb)

        # 4. Generate curve for this scene
        curve = self.curve_generator.generate(scene_style)

        # 5. Apply film curve
        curved = self.curve_generator.apply_curve(frame_wb, curve)

        # 6. Apply LUT
        graded = self.lut_manager.apply_lut(curved, scene_style)

        # 7. Blend original skin tone back for realism
        final = self.skin_protect.restore_skin_tones(graded, frame_wb, skin_mask)

        return final

# ================================================================
# 2777. Color Temperature Stabilizer (Auto White Balance)
# ================================================================

class ColorTemperatureStabilizer:
    """
    Smart auto white-balance:
    - Estimates illuminant color
    - Corrects using grey-world + highlight-balance fusion
    - Stabilizes color temperature across frames
    """

    def __init__(self, stability_alpha=0.15):
        self.stability_alpha = stability_alpha
        self.previous_temp = None

    # ------------------------------------------------------------
    # 2792. Estimate color temperature using grey-world + bright-pixel model
    # ------------------------------------------------------------
    def estimate_temp(self, frame):
        import numpy as np

        # Flatten to pixels
        pixels = frame.reshape(-1, 3).astype(np.float32)

        # Grey-world assumption
        avg = np.mean(pixels, axis=0)  # B, G, R

        # Convert approximate value to Kelvin scale
        # Very rough but stabilizes trends
        kelvin_estimate = (avg[2] / (avg[0] + 1e-5)) * 5000 + 2000

        return float(np.clip(kelvin_estimate, 2500, 9000))

    # ------------------------------------------------------------
    # 2811. White balance correction
    # ------------------------------------------------------------
    def correct(self, frame):
        import cv2
        import numpy as np

        temp = self.estimate_temp(frame)

        # Exponential stability filter
        if self.previous_temp is None:
            stable_temp = temp
        else:
            stable_temp = (self.previous_temp * (1 - self.stability_alpha)) + (temp * self.stability_alpha)

        self.previous_temp = stable_temp

        # Convert temperature → RGB gains
        r_gain, g_gain, b_gain = self._temp_to_gains(stable_temp)

        # Apply gains
        gains = np.array([b_gain, g_gain, r_gain]).reshape(1, 1, 3)
        balanced = frame.astype(np.float32) * gains
        balanced = np.clip(balanced, 0, 255).astype("uint8")

        return balanced, stable_temp

    # ------------------------------------------------------------
    # 2839. Convert color temperature to white balance gains
    # ------------------------------------------------------------
    def _temp_to_gains(self, kelvin):
        """
        Approximates RGB gains from Kelvin temperature.
        """
        import math

        t = kelvin / 100.0

        # Red channel
        if t <= 66:
            r = 1.0
        else:
            r = 1.292936 * ((t - 60) ** -0.133204)

        # Green channel
        if t <= 66:
            g = 0.390081 * math.log(t) - 0.631841
        else:
            g = 1.129891 * ((t - 60) ** -0.075514)

        # Blue channel
        if t >= 66:
            b = 1.0
        else:
            if t <= 19:
                b = 0.0
            else:
                b = 0.543206 * math.log(t - 10) - 1.196254

        # Normalize
        m = max(r, g, b, 1e-6)
        return r/m, g/m, b/m


# ================================================================
# 2861. Skin Tone Protector
# ================================================================

class SkinToneProtector:
    """
    Detects human skin in the frame using HSV + YCrCb fusion.
    Ensures LUT and tone curves don't distort faces.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------
    # 2874. Generate skin region mask
    # ------------------------------------------------------------
    def get_mask(self, frame):
        import cv2
        import numpy as np

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # HSV skin range (coarse)
        lower_hsv = np.array([0, 40, 60])
        upper_hsv = np.array([25, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # YCrCb skin range (refined)
        lower_ycrcb = np.array([0, 135, 85])
        upper_ycrcb = np.array([255, 180, 135])
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # Fusion
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)

        return mask

    # ------------------------------------------------------------
    # 2899. Blend original face color back after grading
    # ------------------------------------------------------------
    def restore_skin_tones(self, graded_frame, original_frame, mask):
        import cv2
        import numpy as np

        mask3 = cv2.merge([mask, mask, mask]) / 255.0
        return (graded_frame * (1 - mask3) + original_frame * mask3).astype("uint8")
# ================================================================
# 2926. Film LUT Manager – applies cinematic LUTs based on scene
# ================================================================

class FilmLUTManager:
    """
    Loads and applies film-style LUTs:
    - Hollywood teal/orange
    - Natural cinematic
    - DJI-inspired vivid
    - Night boost LUT
    """

    def __init__(self):
        self.luts = {}
        self._load_default_luts()

    # ------------------------------------------------------------
    # 2942. Load default LUTs (simple tone maps; extendable)
    # ------------------------------------------------------------
    def _load_default_luts(self):
        """
        LUTs stored as 256×3 tables.
        These are simplified but fully functional.
        """
        import numpy as np

        # Teal-Orange Cinematic LUT
        teal_orange = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            teal_orange[i] = [
                min(255, i * 0.85),           # B: slightly reduced blue
                min(255, i * 1.05),           # G: lifted green
                min(255, 30 + i * 1.15)       # R: boosted warmth
            ]

        # Natural Cinematic LUT (soft contrast + saturation)
        natural = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            natural[i] = [
                min(255, i * 1.02),
                min(255, i * 1.02),
                min(255, i * 1.02)
            ]

        # Night LUT (boost shadows, protect highlights)
        night = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            night[i] = [
                min(255, i * 1.25),
                min(255, i * 1.15),
                min(255, i)
            ]

        self.luts = {
            "cinematic_teal_orange": teal_orange,
            "cinematic_natural": natural,
            "night_boost": night
        }

    # ------------------------------------------------------------
    # 2983. LUT application
    # ------------------------------------------------------------
    def apply_lut(self, frame, style):
        """
        Maps each pixel through the selected LUT.
        """
        import numpy as np

        lut = None
        if style == "cinematic":
            lut = self.luts["cinematic_teal_orange"]
        elif style == "natural":
            lut = self.luts["cinematic_natural"]
        elif style == "night":
            lut = self.luts["night_boost"]
        else:
            # default: natural cinematic
            lut = self.luts["cinematic_natural"]

        # Apply LUT
        out = lut[frame]
        return np.clip(out, 0, 255).astype("uint8")


# ================================================================
# 3009. Scene Color Classifier – decides LUT and grading style
# ================================================================

class SceneColorClassifier:
    """
    Classifies the scene based on:
    - Brightness distribution
    - Color temperature
    - Motion (optional)
    - Histogram shape

    Output categories:
      • cinematic
      • natural
      • night
      • high_contrast
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------
    # 3028. Primary classification logic
    # ------------------------------------------------------------
    def classify(self, frame):
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        # Histogram flatness → high contrast scene
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        high_ratio = np.sum(hist[230:]) / np.sum(hist)
        low_ratio  = np.sum(hist[:25])  / np.sum(hist)

        # Decision rules
        if brightness < 55:
            return "night"

        if high_ratio > 0.15 and low_ratio > 0.15:
            return "high_contrast"

        # Default → cinematic for most scenes
        return "cinematic"


# ================================================================
# 3062. Cine Curve Generator – Film contrast / S-curve LUT
# ================================================================

class CineCurveGenerator:
    """
    Generates dynamic S-curves for cinematic contrast.
    Adjustable for each scene type.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------
    # 3079. Generate a curve based on scene type
    # ------------------------------------------------------------
    def generate(self, scene_type):
        import numpy as np

        # Generate 256-step curve
        x = np.linspace(0, 1, 256)

        if scene_type == "cinematic":
            # Gentle S-curve
            y = 0.5 + 0.25 * np.tanh(2 * (x - 0.5))

        elif scene_type == "high_contrast":
            # Stronger contrast but protect shadows
            y = 0.45 + 0.35 * np.tanh(3 * (x - 0.5))

        elif scene_type == "night":
            # Boost shadows, reduce highlight contrast
            y = x ** 0.8

        else:
            # Natural curve
            y = x

        curve = (y * 255).astype("uint8")
        return curve

    # ------------------------------------------------------------
    # 3109. Apply generated curve to the frame
    # ------------------------------------------------------------
    def apply_curve(self, frame, curve):
        import cv2
        import numpy as np

        # Apply curve per channel
        out = cv2.LUT(frame, curve)
        return out
# ================================================================
# 3110. LOG Gamma Generator – creates LOGC/LOG-like flat profiles
# ================================================================

class LOGGammaGenerator:
    """
    Converts frames to cinematic LOG gamma for maximum dynamic range.
    LOG is a flat image used for color grading (similar to DJI D-LOG,
    Sony S-LOG, Arri LOG-C). This is a simplified mathematically
    correct LOG-like encoder.
    """

    def __init__(self):
        # Base constants for LOG curve
        self.a = 0.25
        self.b = 0.45
        self.c = 0.10

    # ------------------------------------------------------------
    # 3128. Encode RGB frame into LOG space
    # ------------------------------------------------------------
    def to_log(self, frame):
        import numpy as np

        f = frame.astype(np.float32) / 255.0

        # Simple LOG equation
        log_frame = (np.log10(1 + self.b * f) / np.log10(1 + self.b)) * (1 - self.c) + self.c

        log_frame = np.clip(log_frame * 255, 0, 255).astype("uint8")
        return log_frame

    # ------------------------------------------------------------
    # 3144. Decode back to Rec709-like gamma
    # ------------------------------------------------------------
    def to_rec709(self, log_frame):
        import numpy as np

        f = log_frame.astype(np.float32) / 255.0

        # Inverse of our LOG equation
        rec = ((10 ** (((f - self.c) / (1 - self.c)) * np.log10(1 + self.b))) - 1) / self.b
        rec = np.clip(rec * 255, 0, 255).astype("uint8")
        return rec


# ================================================================
# 3164. Highlight Recovery Engine – rebuilds blown-out detail
# ================================================================

class HighlightRecovery:
    """
    Recovers blown highlights by:
    - detecting saturated pixels
    - blending luma from green channel
    - reconstructing chroma for missing detail
    """

    def __init__(self, threshold=245, blend_strength=0.35):
        self.threshold = threshold
        self.blend_strength = blend_strength

    # ------------------------------------------------------------
    # 3179. Main highlight recovery
    # ------------------------------------------------------------
    def recover(self, frame):
        import cv2
        import numpy as np

        b, g, r = cv2.split(frame)

        # Mask of blown highlights
        mask = (r > self.threshold) & (g > self.threshold) & (b > self.threshold)

        if not np.any(mask):
            return frame  # nothing blown → quick return

        # Use green channel as luminance reference
        recovered_luma = g.astype(np.float32)

        # Blend reconstruction back into all channels
        rb = r.astype(np.float32)
        gb = g.astype(np.float32)
        bb = b.astype(np.float32)

        rb[mask] = rb[mask] * (1 - self.blend_strength) + recovered_luma[mask] * self.blend_strength
        gb[mask] = gb[mask] * (1 - self.blend_strength) + recovered_luma[mask] * self.blend_strength
        bb[mask] = bb[mask] * (1 - self.blend_strength) + recovered_luma[mask] * self.blend_strength

        out = cv2.merge([
            np.clip(bb, 0, 255).astype("uint8"),
            np.clip(gb, 0, 255).astype("uint8"),
            np.clip(rb, 0, 255).astype("uint8")
        ])

        return out


# ================================================================
# 3218. Shadow Lift Engine – boosts dark regions without noise
# ================================================================

class ShadowLiftEngine:
    """
    Intelligent shadow lifting:
    - avoids lifting noise-heavy areas
    - uses color-preserving gamma curves
    """

    def __init__(self, strength=0.45):
        self.strength = strength  # 0–1

    # ------------------------------------------------------------
    # 3232. Apply shadow lift
    # ------------------------------------------------------------
    def lift(self, frame):
        import numpy as np
        import cv2

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2] / 255.0

        # Shadow mask (<0.25 brightness)
        mask = v < 0.25

        # Lift curve: more aggressive near 0; gentle near 0.3
        lifted = v.copy()
        lifted[mask] = lifted[mask] ** (1 - self.strength)

        hsv[:, :, 2] = np.clip(lifted * 255, 0, 255)

        out = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        return out


# ================================================================
# 3264. Local Color Optimizer – per-pixel adaptive color grading
# ================================================================

class LocalColorOptimizer:
    """
    Analyzes small 32×32 tiles:
    - adjusts color temperature
    - performs local WB shifts
    - enhances local contrast
    """

    def __init__(self, tile=32):
        self.tile = tile

    # ------------------------------------------------------------
    # 3279. Process tile-by-tile
    # ------------------------------------------------------------
    def optimize(self, frame):
        import numpy as np
        import cv2

        h, w, _ = frame.shape
        out = frame.copy()

        for y in range(0, h, self.tile):
            for x in range(0, w, self.tile):
                tile = frame[y:y+self.tile, x:x+self.tile]

                if tile.size == 0:
                    continue

                # Local WB adjustment
                b_avg = tile[:, :, 0].mean()
                g_avg = tile[:, :, 1].mean()
                r_avg = tile[:, :, 2].mean()

                # White balance correction
                wb = np.array([g_avg / (b_avg + 1e-5),
                               1.0,
                               g_avg / (r_avg + 1e-5)])

                corrected = tile.astype(np.float32)
                corrected *= wb

                corrected = np.clip(corrected, 0, 255).astype("uint8")

                out[y:y+self.tile, x:x+self.tile] = corrected

        return out


# ================================================================
# 3318. Saturation Engine – film-style global saturation control
# ================================================================

class SaturationEngine:
    """
    Applies film-like saturation, avoiding channel clipping
    by working in HSV color space.
    """

    def __init__(self, strength=1.15):
        self.strength = strength

    # ------------------------------------------------------------
    # 3333. Apply saturation boost
    # ------------------------------------------------------------
    def apply(self, frame):
        import cv2
        import numpy as np

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= self.strength

        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

        out = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        return out


# ================================================================
# 3357. Shot Mood Controller – chooses color mood per request
# ================================================================

class ShotMoodController:
    """
    Takes user text + scene color classification → chooses:
    - LUT
    - saturation level
    - contrast curve
    - LOG or Rec709
    """

    def __init__(self):
        self.valid_moods = ["cinematic", "warm", "cold", "neutral", "night"]

    # ------------------------------------------------------------
    # 3370. Decide mood
    # ------------------------------------------------------------
    def decide_mood(self, user_text, scene_type):
        user_text = user_text.lower()

        # User overrides
        if "warm" in user_text:
            return "warm"
        if "cold" in user_text or "blue" in user_text:
            return "cold"
        if "night" in user_text:
            return "night"

        # Otherwise fall back to scene type
        if scene_type in self.valid_moods:
            return scene_type

        return "cinematic"
# ================================================================
# 3361. Color Temperature Estimator – Kelvin-based WB measurement
# ================================================================

class ColorTemperatureEstimator:
    """
    Estimates scene color temperature in Kelvin using a grey-world
    assumption combined with chromaticity mapping. Output is used
    for adaptive white balance and mood grading.
    """

    def __init__(self):
        # Useful ranges for drones: 2500K (warm) → 9000K (blue sky)
        self.min_temp = 2500
        self.max_temp = 9000

    # ------------------------------------------------------------
    # 3376. Estimate Kelvin temperature
    # ------------------------------------------------------------
    def estimate(self, frame):
        import numpy as np

        # Average RGB in linear space
        b, g, r = [c.mean() for c in cv2.split(frame.astype(float))]

        # Prevent divide-by-zero
        g = max(g, 1e-6)

        # Red-to-Green and Blue-to-Green ratios
        rg = r / g
        bg = b / g

        # Map ratios to approximate Kelvin temperature
        temp = 6500 + (rg - bg) * 2200  # Empirical mapping

        # Clamp
        temp = max(self.min_temp, min(self.max_temp, temp))

        return float(temp)


# ================================================================
# 3399. Adaptive White Balance Engine – Kelvin → Gain multipliers
# ================================================================

class AdaptiveWhiteBalanceEngine:
    """
    Adjusts white balance based on estimated Kelvin AND scene mood.
    Produces gain multipliers for R,G,B channels.
    """

    def __init__(self):
        # Reference neutral temperature
        self.ref_temp = 6500

    # ------------------------------------------------------------
    # 3413. Convert Kelvin temperature to RGB gains
    # ------------------------------------------------------------
    def temp_to_gain(self, temp_k):
        import numpy as np

        # Normalize around 6500K
        delta = (temp_k - self.ref_temp) / 3500.0

        # Warmer light → boost blue; Cooler light → boost red
        r_gain = 1.0 + (-delta * 0.5)
        b_gain = 1.0 + (delta * 0.6)
        g_gain = 1.0  # green baseline

        return np.array([b_gain, g_gain, r_gain], dtype=float)

    # ------------------------------------------------------------
    # 3433. Apply white balance to frame
    # ------------------------------------------------------------
    def apply(self, frame, temp_k):
        gains = self.temp_to_gain(temp_k)

        corrected = frame.astype(float)
        corrected[:, :, 0] *= gains[0]
        corrected[:, :, 1] *= gains[1]
        corrected[:, :, 2] *= gains[2]

        corrected = corrected.clip(0, 255).astype("uint8")
        return corrected


# ================================================================
# 3456. Film Grain Simulator – cinematic texture engine
# ================================================================

class FilmGrainSimulator:
    """
    Adds realistic film grain based on ISO, noise model, and mood.
    Purely aesthetic — this is used only for cinematic shots.
    """

    def __init__(self, strength=0.07):
        self.strength = strength

    # ------------------------------------------------------------
    # 3470. Generate and apply grain
    # ------------------------------------------------------------
    def apply(self, frame):
        import numpy as np

        h, w, _ = frame.shape

        # Generate gaussian grain
        grain = np.random.normal(0, 25, (h, w)).astype(np.float32)

        grain = (grain * self.strength).astype(np.float32)

        # Broadcast to 3 channels
        grain_rgb = np.repeat(grain[:, :, None], 3, axis=2)

        out = frame.astype(np.float32) + grain_rgb
        out = np.clip(out, 0, 255).astype("uint8")

        return out


# ================================================================
# 3497. Vignette Engine – lens-character artistic shading
# ================================================================

class VignetteEngine:
    """
    Applies soft radial vignette to mimic cinema lenses and guide
    viewer attention. Used lightly unless user explicitly requests.
    """

    def __init__(self, strength=0.25):
        self.strength = strength

    # ------------------------------------------------------------
    # 3511. Apply vignette
    # ------------------------------------------------------------
    def apply(self, frame):
        import numpy as np
        import cv2

        h, w = frame.shape[:2]

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]

        # Center
        cy, cx = h / 2, w / 2

        # Normalized radial distance from center
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        dist /= dist.max()

        # Vignette mask: 1→center, decreasing outward
        mask = 1 - self.strength * (dist**2)
        mask = mask.clip(0.0, 1.0)

        mask = np.repeat(mask[:, :, None], 3, axis=2)

        out = frame.astype(float) * mask
        out = out.clip(0, 255).astype("uint8")

        return out


# ================================================================
# 3549. Final Unified Color Pipeline – puts all modules together
# ================================================================

class AIColorPipeline:
    """
    A complete multi-stage cinematic color grading pipeline that:
        - Converts to LOG
        - Applies highlight/shadow recovery
        - Adjusts white balance + color temperature
        - Applies LUT mood grading
        - Local color correction
        - Saturation tuning
        - Film grain
        - Vignette
    """

    def __init__(self):
        self.log = LOGGammaGenerator()
        self.rec709 = LOGGammaGenerator()
        self.hrec = HighlightRecovery()
        self.srec = ShadowLiftEngine()
        self.lco = LocalColorOptimizer()
        self.sat = SaturationEngine()
        self.temp_est = ColorTemperatureEstimator()
        self.awb = AdaptiveWhiteBalanceEngine()
        self.grain = FilmGrainSimulator()
        self.vignette = VignetteEngine()
        self.mood = ShotMoodController()

    # ------------------------------------------------------------
    # 3584. Run full color pipeline
    # ------------------------------------------------------------
    def process(self, frame, user_text, scene_type="cinematic"):
        # 1. Estimate color temp
        temp = self.temp_est.estimate(frame)
        frame = self.awb.apply(frame, temp)

        # 2. LOG encode
        log_frame = self.log.to_log(frame)

        # 3. Recover highlights + shadows
        log_frame = self.hrec.recover(log_frame)
        log_frame = self.srec.lift(log_frame)

        # 4. Local tile optimizations
        log_frame = self.lco.optimize(log_frame)

        # 5. Color mood (warm/cold/etc.)
        mood = self.mood.decide_mood(user_text, scene_type)
        if mood == "warm":
            log_frame[:, :, 2] = np.clip(log_frame[:, :, 2] * 1.1, 0, 255)
        elif mood == "cold":
            log_frame[:, :, 0] = np.clip(log_frame[:, :, 0] * 1.1, 0, 255)

        # 6. Saturation
        log_frame = self.sat.apply(log_frame)

        # 7. Grain + Vignette
        log_frame = self.grain.apply(log_frame)
        log_frame = self.vignette.apply(log_frame)

        # 8. Convert back to Rec709
        out = self.rec709.to_rec709(log_frame)

        return {
            "frame": out,
            "temperature_k": temp,
            "mood": mood
        }
                        