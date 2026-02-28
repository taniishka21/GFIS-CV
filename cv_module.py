import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import tensorflow as tf

# Load model once at the top
model = tf.keras.models.load_model("waste_mobilenet.h5")
classes = ['agriculture', 'contaminated', 'food', 'manure']

class WasteProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # ROI logic (mirrored from your original code)
        h, w, _ = img.shape
        x1, y1 = int(w*0.3), int(h*0.3)
        x2, y2 = int(w*0.7), int(h*0.7)
        
        roi = img[y1:y2, x1:x2]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Preprocess & Predict
        input_img = cv2.resize(roi, (224, 224)) / 255.0
        input_img = np.expand_dims(input_img, axis=0)
        
        prediction = model.predict(input_img, verbose=0)
        class_id = np.argmax(prediction[0])
        label = classes[class_id]
        conf = float(np.max(prediction[0]))

        # UI Overlay
        color = (0, 0, 255) if label == "contaminated" else (0, 255, 0)
        cv2.putText(img, f"{label.upper()} ({conf:.2f})", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return img

st.title("♻️ GFIS – Live Waste Acceptance")

# This replaces your "Start/Stop" buttons and custom loop
webrtc_streamer(key="waste-check", video_transformer_factory=WasteProcessor)
