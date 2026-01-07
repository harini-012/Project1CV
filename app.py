import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import time
from collections import OrderedDict

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Traffic Analyzer",
    layout="wide",
    page_icon="🚦"
)

# ---------------- Custom CSS for Cards ----------------
st.markdown("""
<style>
.card {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0px;
    text-align: center;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}
.card h3 {
    margin-bottom: 8px;
    color: #222222;
}
.card p {
    color: #555555;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- App Header ----------------
st.title("🚦 Professional Traffic Analyzer")
st.markdown("""
Detect and analyze traffic from images or videos using **AI-powered vehicle detection**.  
View vehicle counts and traffic status in a clean, professional layout.
""")

# ---------------- Feature Cards ----------------
st.markdown("### App Features")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="card">
        <h3>Supported Media</h3>
        <p>Images & Videos (jpg, png, mp4, avi, mov)</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="card">
        <h3>Detectable Vehicles</h3>
        <p>Cars, Trucks, Buses, Motorcycles</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="card">
        <h3>Traffic Analysis</h3>
        <p>Low 🟢 / Moderate 🟡 / High 🔴</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- File Uploader ----------------
uploaded_file = st.file_uploader(
    "Upload a traffic image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# Clear previous results when a new file is uploaded
if "prev_file" not in st.session_state:
    st.session_state.prev_file = None
if uploaded_file is not None:
    if uploaded_file != st.session_state.prev_file:
        st.session_state.prev_file = uploaded_file
        st.session_state.counted_ids = set()  # Reset previous vehicle count

frame_window = st.empty()
results_container = st.container()  # Container to show analysis results

# ---------------- Load YOLOv8 ----------------
model = YOLO("yolov8n.pt")
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

# ---------------- Centroid Tracker ----------------
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = np.linalg.norm(
                np.array(objectCentroids)[:, np.newaxis] - inputCentroids[np.newaxis, :],
                axis=2,
            )
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])) - usedRows
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in set(range(0, D.shape[1])) - usedCols:
                self.register(inputCentroids[col])

        return self.objects

# ---------------- Traffic Status ----------------
def traffic_status(count):
    if count == 0:
        return "❌ No vehicles detected"
    elif count <= 5:
        return "Low Traffic 🟢"
    elif count <= 15:
        return "Moderate Traffic 🟡"
    else:
        return "High Traffic 🔴"

# ---------------- Process Frame ----------------
def process_frame(frame, tracker):
    results = model(frame)[0]
    rects = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        cls_name = model.names[int(cls)]
        if cls_name in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box)
            rects.append((x1, y1, x2, y2))
    objects = tracker.update(rects)

    for objectID, centroid in objects.items():
        cv2.putText(
            frame,
            f"ID {objectID}",
            (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    return frame, len(objects)

# ---------------- Main ----------------
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_ext = uploaded_file.name.split(".")[-1].lower()
    tracker = CentroidTracker()
    st.session_state.counted_ids = set()  # Reset count for this upload

    if file_ext in ["jpg", "jpeg", "png"]:
        np_img = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        processed_img, vehicle_count = process_frame(img, tracker)
        frame_window.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))

        # Show results in cards
        results_container.empty()  # Clear previous results
        col1, col2 = results_container.columns(2)
        with col1:
            st.markdown(f"""
            <div class="card">
                <h3>Vehicle Count</h3>
                <p>{vehicle_count}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="card">
                <h3>Traffic Status</h3>
                <p>{traffic_status(vehicle_count)}</p>
            </div>
            """, unsafe_allow_html=True)

    elif file_ext in ["mp4", "avi", "mov"]:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file_bytes)
        cap = cv2.VideoCapture(tfile.name)
        frame_counter = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        results_container.empty()  # Clear previous results

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, vehicle_count = process_frame(frame, tracker)
            for objectID in tracker.objects.keys():
                st.session_state.counted_ids.add(objectID)

            frame_window.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            frame_counter += 1
            progress_bar.progress(frame_counter / total_frames)
            time.sleep(0.01)

        cap.release()
        col1, col2 = results_container.columns(2)
        with col1:
            st.markdown(f"""
            <div class="card">
                <h3>Total Unique Vehicles</h3>
                <p>{len(st.session_state.counted_ids)}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="card">
                <h3>Traffic Status</h3>
                <p>{traffic_status(len(st.session_state.counted_ids))}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    frame_window.empty()
    results_container.empty()
    st.warning("Please upload a traffic image or video to start analysis.")
