import csv
import time
import os

import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from features import extract_features


MODEL_PATH = "hand_landmarker.task"
DATASET_PATH = "dataset.csv"

LABEL_KEYS = {
    ord("1"): "rock",
    ord("2"): "paper",
    ord("3"): "scissors",
    ord("4"): "thumb_up",
    ord("5"): "ok",
    ord("0"): "none",
}


# ============================
# MediaPipe
# ============================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(options)


# ============================
# RealSense
# ============================

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


current_label = None
recording = False
samples_collected = 0

print("""
控制：

1 = rock
2 = paper
3 = scissors
4 = thumb_up
5 = ok
0 = none

SPACE = 开始/暂停采集
s     = 完成当前动作采集并清空标签
q     = 退出

使用流程：选标签(数字) → 空格开始 → 做手势 → s 完成
""")


file_exists = os.path.exists(DATASET_PATH)

f = open(DATASET_PATH, "a", newline="")
writer = csv.writer(f)

if not file_exists:
    writer.writerow(
        ["label"] +
        [f"f{i}" for i in range(63)]
    )


try:

    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(time.monotonic() * 1000)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ------------------------
        # 检测到手
        # ------------------------

        if result.hand_world_landmarks:

            landmarks = result.hand_world_landmarks[0]

            features = extract_features(
                landmarks
            )

            if recording and current_label:
                writer.writerow(
                    [current_label] +
                    features.tolist()
                )

                f.flush()
                samples_collected += 1

            cv2.putText(
                frame,
                "HAND FOUND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO HAND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        if current_label is None:
            status_text = "NO LABEL SELECTED"
            status_color = (0, 0, 255)
            hint_text = "Press 1-5 / 0 to select gesture"
        elif recording:
            status_text = f"LABEL: {current_label} | RECORDING {samples_collected}"
            status_color = (0, 255, 0)
            hint_text = "Press SPACE to pause, s to finish"
        else:
            status_text = f"LABEL: {current_label} | PAUSED"
            status_color = (0, 255, 255)
            hint_text = "Press SPACE to resume, s to finish"

        cv2.putText(
            frame,
            status_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

        cv2.putText(
            frame,
            hint_text,
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2
        )

        if recording:
            cv2.circle(
                frame,
                (600, 30),
                12,
                (0, 0, 255),
                -1
            )
            cv2.putText(
                frame,
                "REC",
                (570, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Gesture Dataset Collector",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key in LABEL_KEYS:
            current_label = LABEL_KEYS[key]
            samples_collected = 0

            if recording:
                recording = False
                print(f"[标签切换] 切换为 {current_label}，已自动停止采集。按 SPACE 重新开始。")
            else:
                print(f"[选择标签] {current_label}，按 SPACE 开始采集。")

        elif key == ord(" "):
            if current_label is None:
                print("[提示] 请先按 1-5 或 0 选择一个手势标签，再按空格开始采集。")
            else:
                recording = not recording

                if recording:
                    if samples_collected == 0:
                        print(f"[开始采集] 标签: {current_label}")
                    else:
                        print(f"[继续采集] 标签: {current_label}，已从第 {samples_collected} 帧继续。")
                else:
                    print(f"[暂停采集] 标签: {current_label}，当前已保存 {samples_collected} 帧。")

        elif key == ord("s"):
            if recording and current_label:
                print(f"[完成采集] 动作: {current_label}，共保存 {samples_collected} 帧。请重新选择标签。")
                recording = False
                current_label = None
                samples_collected = 0
            else:
                print("[提示] 当前没有在采集，无需完成。")

        elif key == ord("q"):
            print("[退出程序] 正在保存并关闭相机...")
            break


finally:

    print("[退出程序] 保存文件并释放资源。")
    f.close()
    pipeline.stop()
    landmarker.close()
    cv2.destroyAllWindows()
    print("[退出程序] 已安全退出。")