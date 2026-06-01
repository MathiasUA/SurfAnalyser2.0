import cv2
import subprocess
import sys
import csv
import os
import time
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

YOUTUBE_URL = "https://www.youtube.com/watch?v=Q6eZVkUKFxo"
MODEL_PATH = "yolov8s.pt"

CSV_FILE = "surf_data.csv"
SAVE_INTERVAL_SEC = 5
ANALYZE_INTERVAL_SEC = 0.5

NIGHT_THRESHOLD = 15.0

# -----------------------------------------------------------------------------
# VASTE ZONES (Gekalibreerd voor St. Augustine Beach)
# -----------------------------------------------------------------------------
HORIZON_Y_LEFT = 159
HORIZON_Y_RIGHT = 128

COAST_Y_LEFT = 485
COAST_Y_RIGHT = 375

WATER_ZONE_PTS = np.array([
    [0, 250],
    [572, 253],
    [945, 387],
    [0, 420]
], np.int32)


# =============================================================================
# STREAM HELPERS
# =============================================================================

def get_stream_url(youtube_url: str) -> str:
    cmd = [sys.executable, "-m", "yt_dlp", "-g", youtube_url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp error:\n{result.stderr}")
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("http"):
            return line
    raise RuntimeError("No valid stream URL found.")


# =============================================================================
# DATA SMOOTHING & WAVE DETECTION
# =============================================================================

class DataSmoother:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.history = []

    def add(self, value):
        self.history.append(value)
        if len(self.history) > self.buffer_size:
            self.history.pop(0)

    def get_smoothed(self):
        if not self.history: return 0
        return int(np.median(self.history))


class WaveDetector:
    """Zoekt naar pieken in de hoeveelheid schuim om de golffrequentie te bepalen."""

    def __init__(self):
        self.foam_history = []
        self.peak_times = []

    def update(self, current_foam_percent):
        self.foam_history.append(current_foam_percent)
        # Bewaar de laatste paar seconden aan data (om pieken te zoeken)
        if len(self.foam_history) > 10:
            self.foam_history.pop(0)

        # Piek detectie: Is de middelste meting in de laatste 3 metingen het hoogst?
        if len(self.foam_history) >= 3:
            p_prev = self.foam_history[-3]
            p_curr = self.foam_history[-2]
            p_next = self.foam_history[-1]

            # Een piek moet een 'echte' golf zijn (minstens 1.0% schuim)
            if p_curr > p_prev and p_curr > p_next and p_curr > 1.0:
                now = time.time()
                # Zorg dat we niet 2 pieken meten binnen dezelfde golf (minstens 4 sec ertussen)
                if not self.peak_times or (now - self.peak_times[-1]) > 4.0:
                    self.peak_times.append(now)

        # Gooi pieken ouder dan 3 minuten (180 sec) weg
        now = time.time()
        self.peak_times = [t for t in self.peak_times if now - t < 180]

    def get_waves_per_minute(self):
        # We hebben minstens 2 golven nodig om de tijd ertussen te meten
        if len(self.peak_times) < 2:
            return 0.0

        # Bereken de gemiddelde tijd tussen de opgeslagen pieken
        intervals = [self.peak_times[i] - self.peak_times[i - 1] for i in range(1, len(self.peak_times))]
        avg_interval = sum(intervals) / len(intervals)

        if avg_interval == 0: return 0.0
        # Zet seconden-per-golf om naar golven-per-minuut
        return round(60.0 / avg_interval, 1)


# =============================================================================
# MASKS & ANALYSIS
# =============================================================================

def generate_static_masks(width, height):
    sky_mask = np.zeros((height, width), dtype=np.uint8)
    water_mask = np.zeros((height, width), dtype=np.uint8)
    beach_mask = np.zeros((height, width), dtype=np.uint8)

    sky_poly = np.array([[[0, 0], [width, 0], [width, HORIZON_Y_RIGHT], [0, HORIZON_Y_LEFT]]], np.int32)
    water_poly = np.array([WATER_ZONE_PTS], np.int32)
    beach_poly = np.array([[[0, COAST_Y_LEFT], [width, COAST_Y_RIGHT], [width, height], [0, height]]], np.int32)

    cv2.fillPoly(sky_mask, sky_poly, 255)
    cv2.fillPoly(water_mask, water_poly, 255)
    cv2.fillPoly(beach_mask, beach_poly, 255)

    return sky_mask, water_mask, beach_mask


def sky_brightness_percent(frame, sky_mask):
    """Geeft helderheid terug als een percentage (0% pikdonker, 100% fel wit)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = float(cv2.mean(gray, mask=sky_mask)[0])
    return round((mean_val / 255.0) * 100.0, 1)


def measure_foam_percent(frame, water_mask):
    """Geeft schuimdekking terug als een percentage (0% - 100%)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, -15
    )

    mask = cv2.bitwise_and(binary, binary, mask=water_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    white_pixels = cv2.countNonZero(mask_clean)
    water_area = cv2.countNonZero(water_mask)

    percent = (float(white_pixels) / water_area) * 100.0 if water_area > 0 else 0.0
    return round(percent, 2), mask_clean


def count_people(results, water_mask, beach_mask):
    b_count, w_count = 0, 0
    if results[0].boxes is None:
        return 0, 0

    h, w = water_mask.shape
    for box in results[0].boxes:
        if int(box.cls[0]) != 0: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cx = (x1 + x2) // 2
        feet_y = y2

        if 0 <= feet_y < h and 0 <= cx < w:
            if beach_mask[feet_y, cx] > 0:
                b_count += 1
            elif water_mask[feet_y, cx] > 0:
                w_count += 1

    return b_count, w_count


# =============================================================================
# CSV LOGGING & OVERLAY
# =============================================================================

def init_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # Volledig schone, rauwe data-velden
            writer.writerow([
                "timestamp", "status", "people_beach", "people_water",
                "sky_brightness_pct", "foam_coverage_pct", "wave_freq_bpm"
            ])


def log_csv(path, p_beach, p_water, sky_pct, foam_pct, wave_freq, status):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, status, p_beach, p_water, sky_pct, foam_pct, wave_freq])
    print(f"[{ts}] Logged -> Bch:{p_beach} | Sea:{p_water} | Sky:{sky_pct}% | Foam:{foam_pct}% | Waves:{wave_freq}/min")


def draw_static_zones(frame, sky_mask, water_mask, beach_mask):
    overlay = frame.copy()
    overlay[sky_mask > 0] = [200, 200, 0]
    overlay[water_mask > 0] = [200, 0, 0]
    overlay[beach_mask > 0] = [0, 150, 0]
    cv2.polylines(frame, [WATER_ZONE_PTS], True, (255, 255, 0), 2)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)


def draw_hud(frame, p_beach, p_water, sky_pct, foam_pct, wave_freq, status_text):
    h, w = frame.shape[:2]
    pad, line_h = 12, 28
    panel_w, panel_h = 310, 200
    x0, y0 = w - panel_w - 10, 10

    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (255, 255, 255), 1)

    lines = [
        f"Status     : {status_text}",
        f"People Bch : {p_beach}",
        f"People Sea : {p_water}",
        f"Sky Bright : {sky_pct} %",
        f"Foam Area  : {foam_pct} %",
        f"Wave Freq  : {wave_freq} /min",
    ]

    for i, line in enumerate(lines):
        lc = (0, 255, 0)
        if i == 0 and status_text != "ANALYSING": lc = (0, 140, 255)
        cv2.putText(frame, line, (x0 + pad, y0 + pad + (i + 1) * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.60, lc, 1,
                    cv2.LINE_AA)


# =============================================================================
# MAIN
# =============================================================================

def main():
    init_csv(CSV_FILE)

    print("Loading YOLO model (v8s)...")
    model = YOLO(MODEL_PATH)

    print("Fetching stream URL via yt-dlp...")
    stream_url = get_stream_url(YOUTUBE_URL)
    print("Stream URL obtained. Opening capture...")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened(): raise RuntimeError("Could not open stream.")

    print("Stream open. Press 'q' to quit, 'z' to toggle zone overlay.")

    show_zones = True
    last_save_time = time.time()
    last_analyze_time = 0

    masks_generated = False
    sky_mask, water_mask, beach_mask = None, None, None

    has_cache = False
    cache_sky_pct = 0.0
    cache_foam_pct = 0.0
    cache_wave_freq = 0.0
    cache_boxes = []
    cache_foam_mask = None

    smoother_beach = DataSmoother(buffer_size=4)
    smoother_water = DataSmoother(buffer_size=4)
    wave_detector = WaveDetector()

    while True:
        ret, frame = cap.read()
        if not ret: break

        target_width = 1280
        h, w = frame.shape[:2]
        if w != target_width:
            new_height = int(h * (target_width / float(w)))
            frame = cv2.resize(frame, (target_width, new_height))
        else:
            new_height = h

        if not masks_generated:
            sky_mask, water_mask, beach_mask = generate_static_masks(target_width, new_height)
            masks_generated = True

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_night = curr_gray.mean() < NIGHT_THRESHOLD

        display = frame.copy()

        if is_night:
            status_text = "NIGHT_MODE"
            cv2.putText(display, "NIGHT MODE ACTIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            display_p_beach, display_p_water = 0, 0

        else:
            now = time.time()

            if now - last_analyze_time >= ANALYZE_INTERVAL_SEC:
                status_text = "ANALYSING"

                # 1. Bereken % Lucht en % Schuim
                cache_sky_pct = sky_brightness_percent(frame, sky_mask)
                cache_foam_pct, cache_foam_mask = measure_foam_percent(frame, water_mask)

                # 2. Update de Wave Detector met de huidige hoeveelheid schuim
                wave_detector.update(cache_foam_pct)
                cache_wave_freq = wave_detector.get_waves_per_minute()

                # 3. YOLO Detectie
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=0.20,
                                      classes=[0])
                raw_p_beach, raw_p_water = count_people(results, water_mask, beach_mask)

                smoother_beach.add(raw_p_beach)
                smoother_water.add(raw_p_water)

                cache_boxes = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        if int(box.cls[0]) == 0:
                            track_id = int(box.id[0]) if box.id is not None else -1
                            cache_boxes.append(list(map(int, box.xyxy[0].tolist())) + [track_id])

                has_cache = True
                last_analyze_time = now
            else:
                status_text = "TRACKING"

            display_p_beach = smoother_beach.get_smoothed()
            display_p_water = smoother_water.get_smoothed()

            if has_cache:
                display[cache_foam_mask == 255] = [0, 0, 255]

                for box_data in cache_boxes:
                    x1, y1, x2, y2, track_id = box_data
                    cx = (x1 + x2) // 2
                    feet_y = y2

                    col = (255, 255, 255)
                    if 0 <= feet_y < new_height and 0 <= cx < target_width:
                        if beach_mask[feet_y, cx] > 0:
                            col = (0, 255, 0)
                        elif water_mask[feet_y, cx] > 0:
                            col = (255, 0, 0)

                    cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)

            if show_zones:
                draw_static_zones(display, sky_mask, water_mask, beach_mask)

        # Logging
        now = time.time()
        if now - last_save_time >= SAVE_INTERVAL_SEC:
            log_csv(CSV_FILE, display_p_beach, display_p_water, cache_sky_pct, cache_foam_pct, cache_wave_freq,
                    status_text)
            last_save_time = now

        draw_hud(display, display_p_beach, display_p_water, cache_sky_pct, cache_foam_pct, cache_wave_freq, status_text)
        cv2.imshow("Surf Analyser - Sensor Node", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("z"):
            show_zones = not show_zones

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)