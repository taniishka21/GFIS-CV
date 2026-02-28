import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# ---------------- PAGE CONFIG (MUST BE FIRST) ----------------
st.set_page_config(
    page_title="GFIS – Live Waste Detection",
    layout="wide"
)

# ---------------- DARK GFIS THEME ----------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

body {background-color: #0f172a; color: white;}
.block-container {padding-top: 1rem;}
[data-testid="stMetricValue"] {color: #00ff88;}
</style>
""", unsafe_allow_html=True)

st.title("♻️ GFIS – Live Waste Acceptance System")
st.markdown("AI-powered feedstock validation engine")
st.markdown("---")

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("waste_mobilenet.h5")
classes = ['agriculture', 'contaminated', 'food', 'manure']

# ---------------- SESSION STATE ----------------
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

# ---------------- CONTROLS ----------------
col1, col2 = st.columns(2)

if col1.button("▶ Start Camera"):
    st.session_state.camera_running = True

if col2.button("⛔ Stop Camera"):
    st.session_state.camera_running = False

st.markdown("---")

frame_placeholder = st.empty()
status_box = st.empty()
metrics_box = st.empty()

# ---------------- PREDICTION BUFFER ----------------
prediction_buffer = deque(maxlen=10)

# ---------------- CAMERA LOOP FUNCTION ----------------
def run_camera():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Camera not accessible")
        return

    while st.session_state.camera_running:

        ret, frame = cap.read()
        if not ret:
            st.error("❌ Camera error")
            break

        h, w, _ = frame.shape

        # ROI
        x1, y1 = int(w*0.3), int(h*0.3)
        x2, y2 = int(w*0.7), int(h*0.7)

        roi = frame[y1:y2, x1:x2]

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # Preprocess
        img = cv2.resize(roi, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)
        prediction_buffer.append(prediction[0])

        avg_pred = np.mean(prediction_buffer, axis=0)

        class_id = np.argmax(avg_pred)
        confidence = float(np.max(avg_pred))
        label = classes[class_id]

        # Decision
        if label == "contaminated":
            decision = "NOT ACCEPTABLE"
            color = (0,0,255)
            status_box.error(f"❌ NOT ACCEPTABLE ({label.upper()})")
        else:
            decision = "ACCEPTABLE"
            color = (0,255,0)
            status_box.success(f"✅ ACCEPTABLE ({label.upper()})")

        # Overlay text
        cv2.putText(frame,
                    f"{label.upper()} | {decision}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2)

        cv2.putText(frame,
                    f"Confidence: {confidence:.2f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

        # Convert for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

        # Show metrics nicely
        with metrics_box.container():
            c1, c2 = st.columns(2)
            c1.metric("Detected Class", label.upper())
            c2.metric("Confidence", f"{confidence:.2f}")

        time.sleep(0.05)

    cap.release()

# ---------------- RUN CAMERA ----------------
if st.session_state.camera_running:
    run_camera()