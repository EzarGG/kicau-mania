import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame

pygame.mixer.init()
try:
    pygame.mixer.music.load("kicau_mania.mp3")
except:
    print("Peringatan: File kicau_mania.mp3 tidak ditemukan!")


model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2,
                                       min_hand_detection_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cat_video = cv2.VideoCapture("cat_dance.mp4")

show_second_window = False
is_playing = False

WIN_W, WIN_H = 600, 450

print("--- MODE KICAU MANIA + MUSIK AKTIF ---")

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    img = cv2.resize(img, (WIN_W, WIN_H))

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for handLms in result.hand_landmarks:
            
            for lm in handLms:
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

            
            x_pos = handLms[8].x

            if x_pos < 0.3 and not show_second_window:
                show_second_window = True
                if not is_playing:
                    pygame.mixer.music.play(-1)
                    is_playing = True

            if x_pos > 0.7 and show_second_window:
                show_second_window = False
                pygame.mixer.music.stop()
                is_playing = False

    cv2.imshow("Face Cam", img)
    cv2.moveWindow("Face Cam", 50, 150)

    if show_second_window:
        ret_cat, cat_frame = cat_video.read()
        if not ret_cat:
            cat_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_cat, cat_frame = cat_video.read()

        cat_frame = cv2.resize(cat_frame, (WIN_W, WIN_H))
        cv2.imshow("Kucing Joget", cat_frame)
        cv2.moveWindow("Kucing Joget", 50 + WIN_W + 10, 150)
    else:
        try:
            if cv2.getWindowProperty("Kucing Joget", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Kucing Joget")
        except:
            pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cat_video.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
