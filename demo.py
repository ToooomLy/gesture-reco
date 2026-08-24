import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from features import extract_features


MODEL_PATH = "hand_landmarker.task"

classifier = joblib.load(
    "gesture_svm.joblib"
)


# =========================
# MediaPipe
# =========================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(
    options
)


# =========================
# D455
# =========================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

pipeline.start(config)


try:

    while True:

        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(
            color_frame.get_data()
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(
            time.monotonic() * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        text = "No hand"

        if result.hand_world_landmarks:

            landmarks = \
                result.hand_world_landmarks[0]

            x = extract_features(
                landmarks
            ).reshape(1, -1)

            probs = classifier.predict_proba(x)[0]

            index = np.argmax(probs)

            label = classifier.classes_[index]
            confidence = probs[index]

            # 自定义拒识
            if confidence < 0.75:
                label = "none"

            text = (
                f"{label}: "
                f"{confidence:.2f}"
            )

            print(text)

        cv2.putText(
            frame,
            text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Custom Gesture",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:

    pipeline.stop()
    landmarker.close()
    cv2.destroyAllWindows()