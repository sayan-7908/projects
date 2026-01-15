import cv2
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------- Pycaw volume setup ----------
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume  # new pycaw usage
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]
volBar, volPer = 400, 0

# ---------- Mediapipe Tasks setup ----------
# 1) age model download koro:
#   https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task
# 2) tarpor ei file-ta rakho:  ./models/hand_landmarker.task

model_path = "models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1,
    running_mode=VisionRunningMode.VIDEO,
)

# ---------- Webcam ----------
wcam, hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

with HandLandmarker.create_from_options(options) as landmarker:
    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1

        # Mediapipe image banano (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # detect_for_video: video mode e frame index dite hoy
        result = landmarker.detect_for_video(mp_image, frame_idx)

        h, w, c = frame.shape
        lmlist = []

        if result.hand_landmarks:
            # sudhu first hand use korchi
            hand = result.hand_landmarks[0]
            for id, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # Thumb–index distance → volume
        if len(lmlist) != 0:
            x1, y1 = lmlist[4][1], lmlist[4][2]   # thumb tip
            x2, y2 = lmlist[8][1], lmlist[8][2]   # index tip

            cv2.circle(frame, (x1, y1), 10, (255, 255, 255), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 255, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            length = math.hypot(x2 - x1, y2 - y1)

            if length < 50:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # map distance to volume (length 50–220 → dB)
            vol = np.interp(length, [50, 220], [minVol, maxVol])
            volume.SetMasterVolumeLevel(vol, None)

            volBar = np.interp(length, [50, 220], [400, 100])
            volPer = np.interp(length, [50, 220], [0, 100])

            # volume bar
            cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 0), 3)
            cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
            cv2.putText(
                frame,
                f"{int(volPer)}%",
                (40, 450),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 0),
                3,
            )

        cv2.putText(
            frame,
            "Press q to quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Hand Volume (Tasks API)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
