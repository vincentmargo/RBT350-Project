import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class HandDetection:
    """2D hand detection result in pixel coordinates."""

    x_px: int
    y_px: int
    score: float


@dataclass(frozen=True)
class DotDetection:
    """2D dot detection result in pixel coordinates."""

    x_px: int
    y_px: int
    score: float


class HandTracker:
    """
    Webcam hand tracker that returns a single (x,y) point in pixels.

    Preferred backend is MediaPipe Hands (robust). If MediaPipe isn't installed,
    falls back to a simple skin-color segmentation centroid (less robust).
    """

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 1,
        mirror: bool = True,
        point_mode: str = "palm",
    ) -> None:
        self._mirror = bool(mirror)
        self._point_mode = str(point_mode)
        self._last_fps_t = time.time()
        self._fps_ema = None

        self._mp = None
        self._hands = None
        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=int(max_num_hands),
                model_complexity=int(model_complexity),
                min_detection_confidence=float(min_detection_confidence),
                min_tracking_confidence=float(min_tracking_confidence),
            )
        except Exception:
            self._mp = None
            self._hands = None

    @property
    def using_mediapipe(self) -> bool:
        return self._hands is not None

    def _update_fps(self) -> float:
        now = time.time()
        dt = max(1e-6, now - self._last_fps_t)
        self._last_fps_t = now
        fps = 1.0 / dt
        if self._fps_ema is None:
            self._fps_ema = fps
        else:
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * fps
        return float(self._fps_ema)

    def process_bgr(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[HandDetection], float]:
        """
        Returns (frame_bgr_for_display, detection_or_None, fps_ema).
        """
        if frame_bgr is None:
            return frame_bgr, None, self._update_fps()

        frame = frame_bgr
        if self._mirror:
            # Avoid negative strides from ::-1 which OpenCV drawing can't handle.
            frame = np.ascontiguousarray(frame[:, ::-1, :])
        else:
            # Be defensive: some capture backends can produce non-contiguous arrays.
            frame = np.ascontiguousarray(frame)

        det = None
        if self._hands is not None:
            det = self._detect_mediapipe(frame)
        else:
            det = self._detect_skin_centroid(frame)

        fps = self._update_fps()
        return frame, det, fps

    def _detect_mediapipe(self, frame_bgr: np.ndarray) -> Optional[HandDetection]:
        mp = self._mp
        hands = self._hands
        assert mp is not None and hands is not None

        # MediaPipe expects RGB
        frame_rgb = frame_bgr[:, :, ::-1]
        res = hands.process(frame_rgb)
        if not res.multi_hand_landmarks:
            return None

        # Compute a stable point from landmarks.
        h, w = frame_bgr.shape[0], frame_bgr.shape[1]
        lm = res.multi_hand_landmarks[0].landmark
        mode = self._point_mode.lower().strip()
        if mode in ("index_tip", "fingertip", "index"):
            # Index fingertip (landmark 8).
            pts = np.array([[lm[8].x * w, lm[8].y * h]], dtype=float)
            center = pts[0]
        elif mode in ("palm", "center", "mean"):
            # Average of a few palm-ish landmarks (more stable than fingertip).
            ids = [0, 1, 5, 9, 13, 17]
            pts = np.array([[lm[i].x * w, lm[i].y * h] for i in ids], dtype=float)
            center = pts.mean(axis=0)
        else:
            # Default fallback.
            ids = [0, 9]
            pts = np.array([[lm[i].x * w, lm[i].y * h] for i in ids], dtype=float)
            center = pts.mean(axis=0)
        x = int(np.clip(round(center[0]), 0, w - 1))
        y = int(np.clip(round(center[1]), 0, h - 1))
        score = 1.0
        return HandDetection(x_px=x, y_px=y, score=score)

    def _detect_skin_centroid(self, frame_bgr: np.ndarray) -> Optional[HandDetection]:
        # Lightweight fallback: YCrCb skin threshold + largest contour centroid.
        import cv2  # local import so module can be imported without cv2 installed

        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        mask = cv2.medianBlur(mask, 7)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        if area < 2500.0:
            return None
        # Reject huge blobs (often background / lighting) that fill most of the frame.
        h, w = frame_bgr.shape[0], frame_bgr.shape[1]
        if area > 0.60 * float(w * h):
            return None
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 40 or bh < 40:
            return None
        aspect = float(bw) / float(bh + 1e-9)
        if aspect < 0.2 or aspect > 5.0:
            return None
        M = cv2.moments(c)
        if abs(M.get("m00", 0.0)) < 1e-9:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))
        score = float(np.clip(area / (w * h), 0.0, 1.0))
        return HandDetection(x_px=cx, y_px=cy, score=score)

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        det: Optional[HandDetection],
        target_xyz: Optional[np.ndarray] = None,
        *,
        status: str = "",
        fps: Optional[float] = None,
    ) -> np.ndarray:
        import cv2

        # Ensure OpenCV gets a writeable contiguous buffer.
        out = np.ascontiguousarray(frame_bgr)
        if det is not None:
            cv2.circle(out, (int(det.x_px), int(det.y_px)), 10, (0, 255, 0), 2)
            cv2.circle(out, (int(det.x_px), int(det.y_px)), 2, (0, 255, 0), -1)

        y = 24
        backend = "mediapipe" if self.using_mediapipe else "skin-fallback"
        lines = [
            f"backend: {backend}",
            f"status: {status}" if status else "",
            f"fps: {fps:.1f}" if fps is not None else "",
        ]
        if target_xyz is not None:
            t = np.asarray(target_xyz, dtype=float).reshape(3)
            lines.append(f"target xyz (m): [{t[0]: .3f}, {t[1]: .3f}, {t[2]: .3f}]")

        for line in lines:
            if not line:
                continue
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y += 22

        return out


class RedDotTracker:
    """Tracks a red dot in the image (intended for red-on-white backgrounds)."""

    def __init__(
        self,
        *,
        mirror: bool = True,
        min_area_px: float = 80.0,
        max_area_frac: float = 0.10,
        min_circularity: float = 0.35,
        dark_threshold: int = -1,
    ) -> None:
        self._mirror = bool(mirror)
        self._min_area_px = float(min_area_px)
        self._max_area_frac = float(max_area_frac)
        self._min_circularity = float(min_circularity)
        self._dark_threshold = int(dark_threshold)
        self._last_fps_t = time.time()
        self._fps_ema = None
        self._last_mask = None

    def _update_fps(self) -> float:
        now = time.time()
        dt = max(1e-6, now - self._last_fps_t)
        self._last_fps_t = now
        fps = 1.0 / dt
        if self._fps_ema is None:
            self._fps_ema = fps
        else:
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * fps
        return float(self._fps_ema)

    def process_bgr(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[DotDetection], float]:
        if frame_bgr is None:
            return frame_bgr, None, self._update_fps()

        frame = frame_bgr
        if self._mirror:
            frame = np.ascontiguousarray(frame[:, ::-1, :])
        else:
            frame = np.ascontiguousarray(frame)

        det = self._detect_red_dot(frame)
        fps = self._update_fps()
        return frame, det, fps

    def get_last_mask_u8(self) -> Optional[np.ndarray]:
        """Returns the most recent binary mask (uint8 0/255) used for detection."""
        return self._last_mask

    def _detect_red_dot(self, frame_bgr: np.ndarray) -> Optional[DotDetection]:
        import cv2

        # Slight blur reduces sensor noise and stabilizes thresholding.
        frame_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

        # Strategy:
        # 1) Try HSV red mask (works when the dot is truly red).
        # 2) Fallback: on bright backgrounds, detect a dark-ish circular blob (works even if
        #    the camera sees the dot as gray due to auto white balance / exposure).
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        # Two ranges because red wraps hue at 0/180 in OpenCV HSV.
        # Widened thresholds to be more tolerant to lighting / screen brightness:
        # - wider hue bands (red can drift toward orange/pink on cameras)
        # - lower S/V minima so dimmer red still counts
        lower1 = np.array([0, 120, 80], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 120, 80], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.medianBlur(mask, 7)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        det = self._pick_contour_centroid(frame_bgr, contours, base_mask=mask)
        if det is not None:
            return det

        return self._detect_dark_blob(frame_bgr)

    def _pick_contour_centroid(
        self, frame_bgr: np.ndarray, contours, *, base_mask: Optional[np.ndarray] = None
    ) -> Optional[DotDetection]:
        import cv2

        if not contours:
            if base_mask is not None:
                self._last_mask = np.zeros_like(base_mask)
            return None

        h, w = frame_bgr.shape[0], frame_bgr.shape[1]
        area_max = float(self._max_area_frac) * float(w * h)

        best = None
        best_score = -1.0
        best_contour = None
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self._min_area_px:
                continue
            if area_max > 0.0 and area > area_max:
                continue
            peri = float(cv2.arcLength(c, True))
            if peri < 1e-6:
                continue
            circularity = float(4.0 * np.pi * area / (peri * peri))
            if circularity < float(self._min_circularity):
                continue
            M = cv2.moments(c)
            if abs(M.get("m00", 0.0)) < 1e-9:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            # Prefer larger + more circular blobs.
            score = float(circularity * area)
            if score > best_score:
                best_score = score
                best = (cx, cy, area)
                best_contour = c

        if best is None:
            if base_mask is not None:
                self._last_mask = np.zeros_like(base_mask)
            return None

        cx, cy, area = best
        cx_i = int(np.clip(round(cx), 0, w - 1))
        cy_i = int(np.clip(round(cy), 0, h - 1))
        score = float(np.clip(area / (w * h), 0.0, 1.0))

        # For visualization, store a mask that only includes the accepted contour.
        if base_mask is not None and best_contour is not None:
            kept = np.zeros_like(base_mask)
            cv2.drawContours(kept, [best_contour], -1, 255, thickness=-1)
            self._last_mask = kept

        return DotDetection(x_px=cx_i, y_px=cy_i, score=score)

    def _detect_dark_blob(self, frame_bgr: np.ndarray) -> Optional[DotDetection]:
        import cv2

        frame_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)

        # Threshold: if dark_threshold < 0 use Otsu, else fixed threshold.
        if int(self._dark_threshold) < 0:
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            thr = int(np.clip(int(self._dark_threshold), 0, 255))
            _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

        mask = cv2.medianBlur(mask, 7)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._pick_contour_centroid(frame_bgr, contours, base_mask=mask)

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        det: Optional[DotDetection],
        target_xyz: Optional[np.ndarray] = None,
        *,
        status: str = "",
        fps: Optional[float] = None,
    ) -> np.ndarray:
        import cv2

        out = np.ascontiguousarray(frame_bgr)
        if det is not None:
            cv2.circle(out, (int(det.x_px), int(det.y_px)), 12, (0, 0, 255), 2)
            cv2.circle(out, (int(det.x_px), int(det.y_px)), 3, (0, 0, 255), -1)

        y = 24
        lines = [
            "backend: red_dot_hsv",
            f"status: {status}" if status else "",
            f"fps: {fps:.1f}" if fps is not None else "",
        ]
        if target_xyz is not None:
            t = np.asarray(target_xyz, dtype=float).reshape(3)
            lines.append(f"target xyz (m): [{t[0]: .3f}, {t[1]: .3f}, {t[2]: .3f}]")

        for line in lines:
            if not line:
                continue
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y += 22

        return out

