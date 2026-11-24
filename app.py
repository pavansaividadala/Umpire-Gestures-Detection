import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import cv2
import tempfile
import os
import io
from collections import Counter, defaultdict

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="UmpireVision AI | Gesture Detection",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best.pt"

# Your trained model contains these four classes:
# DRS, LegBye, NoBall, WideBall

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .main-title {
        font-size: 2.35rem;
        font-weight: 800;
        color: #f4f7fb;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #a8b6c8;
        margin-bottom: 1.8rem;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e8eef7;
        margin: 1rem 0;
    }
    .metric-card {
        background: #1b2635;
        border: 1px solid #304158;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .metric-label {
        color: #a8b6c8;
        font-size: 0.95rem;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 5px;
    }
    div.stButton > button {
        border-radius: 9px;
        font-weight: 650;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🏏 UmpireVision AI</div>'
    '<div class="sub-title">YOLO11-powered cricket umpire signal detection</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

if not Path(MODEL_PATH).is_file():
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Place best (1).pt in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the YOLO model: {error}")
    st.info("Install the required packages and check that the model file is valid.")
    st.stop()

# Display the class names stored in the trained model
class_names = model.names
if isinstance(class_names, dict):
    class_labels = [class_names[key] for key in sorted(class_names)]
else:
    class_labels = list(class_names)

st.caption("Model classes: " + " • ".join(class_labels))

# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------
st.sidebar.header("⚙️ Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Higher values show only more confident detections."
)

st.sidebar.markdown("---")
st.sidebar.write("**Model:** best (1).pt")
st.sidebar.write("**Architecture:** YOLO11")
st.sidebar.write(f"**Classes:** {len(class_labels)}")

# --------------------------------------------------
# INSPECTION MODE SELECTION
# --------------------------------------------------
st.markdown(
    '<div class="section-title">🔍 Choose Detection Mode</div>',
    unsafe_allow_html=True
)

mode = st.radio(
    "Select detection type",
    ["🖼️ Image Detection", "🎬 Video Detection"],
    horizontal=True,
    label_visibility="collapsed"
)

# ==================================================
# IMAGE DETECTION
# ==================================================
if mode == "🖼️ Image Detection":

    st.markdown(
        '<div class="section-title">📤 Upload Cricket Image</div>',
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader(
        "Choose an image containing an umpire signal",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_image is not None:
        original_image = Image.open(uploaded_image).convert("RGB")

        st.image(
            original_image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("🔍 Detect Umpire Signals", use_container_width=True):
            with st.spinner("Detecting umpire signals..."):
                results = model.predict(
                    source=original_image,
                    conf=confidence,
                    verbose=False
                )

                result = results[0]
                annotated_image = Image.fromarray(
                    cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                )

                detections = []

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        class_name = model.names[class_id]
                        score = float(box.conf[0].item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        detections.append({
                            "Umpire Signal": class_name,
                            "Confidence": round(score, 4),
                            "X1": round(x1, 1),
                            "Y1": round(y1, 1),
                            "X2": round(x2, 1),
                            "Y2": round(y2, 1)
                        })

                detection_df = pd.DataFrame(
                    detections,
                    columns=[
                        "Umpire Signal", "Confidence",
                        "X1", "Y1", "X2", "Y2"
                    ]
                )

                st.session_state["image_result"] = {
                    "annotated": annotated_image,
                    "original": original_image,
                    "detections": detection_df
                }

    # Display image results
    if "image_result" in st.session_state:
        image_data = st.session_state["image_result"]

        annotated_image = image_data["annotated"]
        original_image = image_data["original"]
        detection_df = image_data["detections"]

        total_detections = len(detection_df)
        unique_classes = (
            detection_df["Umpire Signal"].nunique()
            if total_detections > 0 else 0
        )
        average_confidence = (
            detection_df["Confidence"].mean()
            if total_detections > 0 else 0
        )

        st.divider()
        st.markdown(
            '<div class="section-title">📊 Image Detection Summary</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Detections</div>
                    <div class="metric-value">{total_detections}</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Signal Types</div>
                    <div class="metric-value">{unique_classes}</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Confidence</div>
                    <div class="metric-value">{average_confidence:.1%}</div>
                </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown(
            '<div class="section-title">🖼️ Visual Detection</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(original_image, use_container_width=True)

        with col2:
            st.markdown("**Detected Umpire Signals**")
            st.image(annotated_image, use_container_width=True)

        st.divider()
        st.markdown(
            '<div class="section-title">🔎 Detection Details</div>',
            unsafe_allow_html=True
        )

        if total_detections > 0:
            st.dataframe(
                detection_df,
                use_container_width=True,
                hide_index=True
            )

            signal_counts = (
                detection_df["Umpire Signal"]
                .value_counts()
                .rename_axis("Umpire Signal")
                .reset_index(name="Count")
            )

            st.markdown("**Signal Distribution**")
            st.bar_chart(
                signal_counts,
                x="Umpire Signal",
                y="Count"
            )
        else:
            st.success(
                "No umpire signals detected above the selected confidence threshold."
            )

        image_buffer = io.BytesIO()
        annotated_image.save(image_buffer, format="PNG")

        st.download_button(
            label="⬇️ Download Annotated Image",
            data=image_buffer.getvalue(),
            file_name="umpire_signal_detection.png",
            mime="image/png",
            use_container_width=True
        )

# ==================================================
# VIDEO DETECTION
# ==================================================
elif mode == "🎬 Video Detection":

    st.markdown(
        '<div class="section-title">🎬 Upload Cricket Video</div>',
        unsafe_allow_html=True
    )

    uploaded_video = st.file_uploader(
        "Choose a cricket video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        st.info(
            "The model will inspect the video frame by frame. "
            "Processing time depends on video length, resolution, and hardware."
        )

        if st.button("🎯 Detect Signals in Video", use_container_width=True):

            input_path = None
            output_path = None
            cap = None
            writer = None

            try:
                # Save uploaded video temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(uploaded_video.name).suffix
                ) as input_temp:
                    input_path = input_temp.name
                    input_temp.write(uploaded_video.getvalue())

                # Open input video
                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    st.error("Unable to open this video. Please try another file.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if fps <= 0:
                    fps = 25.0

                if width <= 0 or height <= 0:
                    st.error("Invalid video dimensions.")
                    st.stop()

                # Create output video
                output_temp = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )
                output_path = output_temp.name
                output_temp.close()

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                if not writer.isOpened():
                    st.error("Unable to create the output video.")
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()

                frame_number = 0
                total_detections = 0
                class_counts = Counter()
                class_confidences = defaultdict(list)

                with st.spinner("Processing video frames..."):
                    while True:
                        success, frame = cap.read()

                        if not success:
                            break

                        frame_number += 1

                        results = model.predict(
                            source=frame,
                            conf=confidence,
                            verbose=False
                        )

                        result = results[0]
                        annotated_frame = result.plot()

                        if result.boxes is not None:
                            for box in result.boxes:
                                class_id = int(box.cls[0].item())
                                class_name = model.names[class_id]
                                score = float(box.conf[0].item())

                                total_detections += 1
                                class_counts[class_name] += 1
                                class_confidences[class_name].append(score)

                        writer.write(annotated_frame)

                        if total_frames > 0:
                            progress_bar.progress(
                                min(frame_number / total_frames, 1.0)
                            )

                        if frame_number % 10 == 0:
                            status_text.write(
                                f"Processed {frame_number} of "
                                f"{total_frames if total_frames > 0 else 'unknown'} frames..."
                            )

                cap.release()
                cap = None
                writer.release()
                writer = None

                progress_bar.progress(1.0)
                status_text.success(
                    f"Video processing completed! Frames processed: {frame_number}"
                )

                with open(output_path, "rb") as video_file:
                    output_video_bytes = video_file.read()

                average_confidence = (
                    sum(
                        sum(scores)
                        for scores in class_confidences.values()
                    ) / total_detections
                    if total_detections > 0 else 0
                )

                video_summary = []
                for class_name, count in class_counts.items():
                    scores = class_confidences[class_name]
                    video_summary.append({
                        "Umpire Signal": class_name,
                        "Detection Count": count,
                        "Average Confidence": round(
                            sum(scores) / len(scores), 4
                        )
                    })

                video_df = pd.DataFrame(
                    video_summary,
                    columns=[
                        "Umpire Signal",
                        "Detection Count",
                        "Average Confidence"
                    ]
                )

                st.session_state["video_result"] = {
                    "video_bytes": output_video_bytes,
                    "total_detections": total_detections,
                    "total_frames": frame_number,
                    "unique_classes": len(class_counts),
                    "average_confidence": average_confidence,
                    "details": video_df
                }

            except Exception as error:
                st.error(f"Video processing error: {error}")

            finally:
                if cap is not None:
                    cap.release()

                if writer is not None:
                    writer.release()

                if input_path and os.path.exists(input_path):
                    os.remove(input_path)

                if output_path and os.path.exists(output_path):
                    os.remove(output_path)

    # Display video results
    if "video_result" in st.session_state:
        video_data = st.session_state["video_result"]

        st.divider()
        st.markdown(
            '<div class="section-title">📊 Video Detection Summary</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        metrics = [
            ("Total Detections", video_data["total_detections"]),
            ("Frames Processed", video_data["total_frames"]),
            ("Signal Types", video_data["unique_classes"]),
            ("Average Confidence", f'{video_data["average_confidence"]:.1%}')
        ]

        for column, (label, value) in zip(
            [col1, col2, col3, col4], metrics
        ):
            with column:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown(
            '<div class="section-title">🎥 Processed Video</div>',
            unsafe_allow_html=True
        )

        st.video(video_data["video_bytes"])

        st.download_button(
            label="⬇️ Download Annotated Video",
            data=video_data["video_bytes"],
            file_name="umpire_signal_detection.mp4",
            mime="video/mp4",
            use_container_width=True
        )

        st.divider()
        st.markdown(
            '<div class="section-title">🔎 Video Detection Details</div>',
            unsafe_allow_html=True
        )

        if not video_data["details"].empty:
            st.dataframe(
                video_data["details"],
                use_container_width=True,
                hide_index=True
            )

            st.markdown("**Signal Distribution Across Frames**")
            st.bar_chart(
                video_data["details"],
                x="Umpire Signal",
                y="Detection Count"
            )
        else:
            st.success(
                "No umpire signals detected in the uploaded video "
                "above the selected confidence threshold."
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#A8BED8; padding:15px;">
        <p>🏏 UmpireVision AI | YOLO11 Object Detection</p>
        <p>AI-powered cricket umpire signal recognition</p>
        <p>Detected classes: DRS • LegBye • NoBall • WideBall</p>
    </div>
    """,
    unsafe_allow_html=True
)
