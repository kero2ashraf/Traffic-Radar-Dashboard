import streamlit as st
import cv2
import tempfile
import os
import math
from datetime import datetime
from ultralytics import YOLO

# ----------------------
# Page config with logo
# ----------------------
st.set_page_config(
    page_title="Traffic Radar Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚗 Traffic Radar Dashboard"  # replace with path to a PNG logo if you have one
)
st.title("🚦 Traffic Radar Dashboard")

# ----------------------
# Backend YOLO model (hidden path)
# ----------------------
MODEL_PATH = r"C:\Users\kirol\OneDrive - Arab Open University - AOU\Desktop\project 2\yolov8n.pt"

@st.cache_resource
def load_model(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return YOLO(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ----------------------
# Sidebar settings
# ----------------------
st.sidebar.header("Radar Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.05)
pixels_per_meter = st.sidebar.number_input("Pixels per meter (Calibration)", min_value=0.1, value=10.0, step=0.1)
speed_threshold_kmph = st.sidebar.number_input("Speed limit (km/h)", min_value=10, max_value=300, value=80)
save_snapshots = st.sidebar.checkbox("Save snapshots for overspeeding vehicles", value=True)

uploaded_video = st.file_uploader("Upload traffic video (mp4, avi, mov)", type=["mp4", "avi", "mov"])
start = st.button("Start Radar")
clear = st.button("Clear outputs")

# ----------------------
# Session state init
# ----------------------
if "flagged_events" not in st.session_state: st.session_state.flagged_events = []
if "detected_table" not in st.session_state: st.session_state.detected_table = []

if clear:
    st.session_state.flagged_events = []
    st.session_state.detected_table = []
    st.success("Cleared all snapshots and events.")

# ----------------------
# Tracker
# ----------------------
class RadarTracker:
    def __init__(self, max_distance=120, stale_frames=300):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance
        self.stale_frames = stale_frames

    def _dist(self, a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def update(self, centers, frame_no):
        assignments = []
        used = set()
        for tid, data in list(self.tracks.items()):
            best_idx = None
            best_d = None
            for i, c in enumerate(centers):
                if i in used: continue
                d = self._dist(data["center"], c)
                if best_d is None or d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx is not None and best_d <= self.max_distance:
                c = centers[best_idx]
                self.tracks[tid]["center"] = c
                self.tracks[tid]["positions"].append((frame_no, c))
                self.tracks[tid]["last_frame"] = frame_no
                assignments.append((tid, c))
                used.add(best_idx)
        for i, c in enumerate(centers):
            if i in used: continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"center": c, "positions": [(frame_no, c)], "last_frame": frame_no, "flagged": False}
            assignments.append((tid, c))
        to_remove = [tid for tid, d in self.tracks.items() if frame_no - d["last_frame"] > self.stale_frames]
        for tid in to_remove: del self.tracks[tid]
        return assignments

    def get_history(self, tid):
        return self.tracks.get(tid, {}).get("positions", [])

    def mark_flagged(self, tid):
        if tid in self.tracks:
            self.tracks[tid]["flagged"] = True

tracker = RadarTracker(max_distance=100)

def speed_from_positions(positions, fps, ppm, smoothing_frames=3):
    if len(positions) < 2: return 0.0
    last_frame, last_pos = positions[-1]
    target_frame = last_frame - smoothing_frames
    earlier = None
    for fr, p in reversed(positions):
        if fr <= target_frame:
            earlier = (fr, p)
            break
    if earlier is None: earlier = positions[0]
    frame_diff = last_frame - earlier[0]
    if frame_diff <= 0: return 0.0
    dx = last_pos[0] - earlier[1][0]
    dy = last_pos[1] - earlier[1][1]
    px_dist = math.hypot(dx, dy)
    meters = px_dist / max(ppm, 1e-6)
    seconds = frame_diff / (fps if fps > 0 else 25.0)
    mps = meters / seconds
    return mps * 3.6

# ----------------------
# Snapshot folder
# ----------------------
snap_dir = "radar_snapshots"
os.makedirs(snap_dir, exist_ok=True)

# ----------------------
# Table placeholder
# ----------------------
table_placeholder = st.empty()

# ----------------------
# Processing
# ----------------------
if uploaded_video is not None and start:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st.write(f"Video resolution: {width}x{height} | FPS: {round(fps,2)}")

    frame_placeholder = st.empty()
    frame_no = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_no += 1

        # YOLO detection
        results = model.predict(frame, conf=confidence, classes=[2,3,5,7])
        r = results[0]

        boxes, centers = [], []
        for box in r.boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
            boxes.append((x1,y1,x2,y2))
            centers.append(((x1+x2)//2, (y1+y2)//2))

        # Tracker update
        assignments = tracker.update(centers, frame_no)
        for idx, (tid, center) in enumerate(assignments):
            history = tracker.get_history(tid)
            speed_kmph = speed_from_positions(history, fps, pixels_per_meter)
            x1,y1,x2,y2 = boxes[idx] if idx < len(boxes) else (0,0,0,0)

            color = (0,255,0)
            label = f"ID:{tid} {int(speed_kmph)} km/h"
            snapshot_path = None

            # Overspeeding
            if speed_kmph >= speed_threshold_kmph and not tracker.tracks[tid]["flagged"]:
                color = (0,0,255)
                label = f"ID:{tid} 🚨 {int(speed_kmph)} km/h"
                tracker.mark_flagged(tid)
                car_crop = frame[y1:y2, x1:x2]
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = os.path.join(snap_dir, f"vehicle_{tid}_{ts}.jpg")
                cv2.imwrite(snapshot_path, car_crop)
                st.session_state.flagged_events.append({"id": tid, "speed": speed_kmph, "snapshot": snapshot_path})
                st.session_state.detected_table.append({"ID": tid, "Speed (km/h)": int(speed_kmph), "Snapshot": snapshot_path})
                st.warning(f"🚨 Vehicle ID {tid} exceeded speed limit ({int(speed_kmph)} km/h)!")
                st.image(car_crop, caption=f"Vehicle ID {tid}", use_container_width=True)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Update live table
        if st.session_state.detected_table:
            df = st.session_state.detected_table.copy()
            table_placeholder.dataframe(df)

        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    cap.release()
    st.success("✅ Radar run completed.")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("Built for: **Traffic Radar Dashboard** | Powered by YOLO & Streamlit | By Eng Kirollos Ashraf")
