"""Live webcam OCR utility using PaddleOCR.

Captures frames from a webcam, runs OCR on a periodic frame interval, and displays
2 windows:
- "Camera Stream": raw camera feed.
- "OCR Result Stream": PaddleOCR overlay image.

Press q to quit.
"""

import os
import time

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV-compatible numpy array."""
    if pil_img.mode == "RGB":
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if pil_img.mode == "RGBA":
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
    if pil_img.mode == "L":
        return np.array(pil_img)

    pil_rgb = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_rgb), cv2.COLOR_RGB2BGR)


def _get_ocr_overlay_image(result_item) -> np.ndarray | None:
    """Extract the OCR visualization image from a PaddleOCR result item."""
    img_data = getattr(result_item, "img", None)
    if isinstance(img_data, dict):
        ocr_img = img_data.get("ocr_res_img")
        if isinstance(ocr_img, Image.Image):
            return pil_to_cv2(ocr_img)

    if isinstance(result_item, dict):
        payload = result_item.get("img")
        if isinstance(payload, dict):
            ocr_img = payload.get("ocr_res_img")
            if isinstance(ocr_img, Image.Image):
                return pil_to_cv2(ocr_img)

    return None


def normalize_ocr_overlay_width(overlay: np.ndarray, reference_width: int) -> np.ndarray:
    """Normalize PaddleOCR overlay when it is returned as a side-by-side composite.

    Some PaddleOCR pipelines return an image composed of 2 panels
    (original/transformed + OCR visualization), which doubles frame width.
    Keep only the right panel so OCR output matches camera frame width.
    """
    if reference_width <= 0:
        return overlay

    overlay_height, overlay_width = overlay.shape[:2]
    if overlay_height <= 0 or overlay_width <= 0:
        return overlay

    # Detect a roughly 2x-wide composite image and keep the OCR panel.
    if (reference_width * 1.9) <= overlay_width <= (reference_width * 2.2):
        midpoint = overlay_width // 2
        return overlay[:, midpoint:]

    return overlay


def configure_camera(cap: cv2.VideoCapture, width: int = 1080, height: int = 720) -> None:
    """Configure webcam resolution and start in autofocus mode."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)


def prepare_frame_for_ocr(frame: np.ndarray, min_width: int | None = None) -> np.ndarray:
    """Return frame for OCR, optionally upscaling when min_width is provided."""
    height, width = frame.shape[:2]
    if min_width is None or min_width <= 0 or width >= min_width:
        return frame

    scale = min_width / width
    target_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)


def scale_frame_for_display(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Resize a frame for display without changing OCR processing size."""
    height, width = frame.shape[:2]
    target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def _get_focus_value(cap: cv2.VideoCapture) -> float | None:
    """Read focus value from camera driver when available."""
    focus = cap.get(cv2.CAP_PROP_FOCUS)
    if not np.isfinite(focus) or focus < 0:
        return None
    return float(focus)


def lock_focus_at_current_value(cap: cv2.VideoCapture) -> tuple[bool, float | None]:
    """Disable autofocus and hold the current focus value.

    Returns whether autofocus was successfully disabled and the focus value used.
    """
    current_focus = _get_focus_value(cap)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    autofocus_disabled = cap.get(cv2.CAP_PROP_AUTOFOCUS) < 0.5

    if current_focus is not None:
        cap.set(cv2.CAP_PROP_FOCUS, current_focus)

    return autofocus_disabled, current_focus


def get_focus_status_text(focus_locked: bool, focus_lock_ok: bool | None, locked_focus: float | None) -> str:
    """Build a short human-readable focus status label."""
    if not focus_locked:
        return "FOCUS: AUTO (PRESS A/C TO LOCK)"
    if focus_lock_ok is False:
        return "FOCUS: LOCK REQUESTED (UNCONFIRMED)"
    if locked_focus is None:
        return "FOCUS: LOCKED"
    return f"FOCUS: LOCKED {locked_focus:.1f}"


def run_camera_ocr(
    camera_index: int = 0,
    ocr_interval_sec: float = 0.35,
    ocr_min_width: int | None = None,
    display_scale: float = 0.5,
    enable_ocr: bool = False,
) -> None:
    """Run live OCR on webcam frames and display two synchronized windows.

    ocr_interval_sec controls how often OCR inference is triggered to keep
    the camera stream responsive.
    ocr_min_width optionally upscales OCR input if frame width is smaller.
    Default is None (no upscaling).
    display_scale controls both preview windows. 0.5 means half-size display.
    enable_ocr controls the initial OCR mode after startup.

    Controls:
    - q: quit
    - a: lock focus once, then run OCR continuously
    - c: lock focus once, then capture OCR only on c key presses
    """
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=True, use_textline_orientation=False)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {camera_index}.")
    configure_camera(cap)

    window_camera = "Camera Stream"
    window_result = "OCR Result Stream"
    cv2.namedWindow(window_camera, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_result, cv2.WINDOW_NORMAL)

    last_ocr_time = 0.0
    last_result_frame: np.ndarray | None = None
    windows_sized = False
    focus_locked = False
    focus_lock_ok: bool | None = None
    locked_focus: float | None = None
    mode = "AUTO" if enable_ocr else "IDLE"
    capture_requested = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if not windows_sized:
                frame_h, frame_w = frame.shape[:2]
                display_w = max(1, int(frame_w * display_scale))
                display_h = max(1, int(frame_h * display_scale))
                cv2.resizeWindow(window_camera, display_w, display_h)
                cv2.resizeWindow(window_result, display_w, display_h)
                windows_sized = True

            camera_preview = frame.copy()
            focus_status = get_focus_status_text(
                focus_locked=focus_locked,
                focus_lock_ok=focus_lock_ok,
                locked_focus=locked_focus,
            )
            mode_status = f"MODE: {mode}"
            cv2.putText(
                camera_preview,
                focus_status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if focus_locked else (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                camera_preview,
                mode_status,
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_camera, scale_frame_for_display(camera_preview, scale=display_scale))

            now = time.monotonic()
            should_run_ocr = False
            if mode == "AUTO" and (now - last_ocr_time) >= ocr_interval_sec:
                should_run_ocr = True
            if mode == "CAPTURE" and capture_requested:
                should_run_ocr = True

            if should_run_ocr:
                ocr_frame = prepare_frame_for_ocr(frame, min_width=ocr_min_width)
                rgb_frame = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2RGB)
                results = ocr.predict(input=rgb_frame)
                last_ocr_time = now
                capture_requested = False

                overlay = None
                if results:
                    overlay = _get_ocr_overlay_image(results[0])
                    if overlay is not None:
                        overlay = normalize_ocr_overlay_width(overlay, reference_width=frame.shape[1])

                if overlay is None:
                    overlay = frame.copy()
                last_result_frame = overlay

            if last_result_frame is not None:
                cv2.imshow(window_result, scale_frame_for_display(last_result_frame, scale=display_scale))
            else:
                cv2.imshow(window_result, scale_frame_for_display(camera_preview, scale=display_scale))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("a"), ord("A")):
                if not focus_locked:
                    focus_lock_ok, locked_focus = lock_focus_at_current_value(cap)
                    focus_locked = True
                mode = "AUTO"
            if key in (ord("c"), ord("C")):
                if not focus_locked:
                    focus_lock_ok, locked_focus = lock_focus_at_current_value(cap)
                    focus_locked = True
                mode = "CAPTURE"
                capture_requested = True
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_ocr(camera_index=1, enable_ocr=False)
