import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Screen size
screen_w, screen_h = pyautogui.size()

# Camera
cap = cv2.VideoCapture(0)

# Smoothening
prev_x, prev_y = 0, 0

# States
last_click_time = 0
dragging = False

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark

            # Index finger (8)
            x, y = lm[8].x, lm[8].y

            curr_x = screen_w * x
            curr_y = screen_h * y

            # Smooth cursor
            smooth_x = prev_x + (curr_x - prev_x) / 5
            smooth_y = prev_y + (curr_y - prev_y) / 5

            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y

            # Distance (index-middle)
            x1, y1 = lm[8].x, lm[8].y
            x2, y2 = lm[12].x, lm[12].y
            dist = math.hypot(x2 - x1, y2 - y1)

            # LEFT CLICK
            if dist < 0.1999:
                current_time = time.time()
                if current_time - last_click_time > 0.5:
                    pyautogui.click()
                    last_click_time = current_time

            # DRAG (hold pinch)
            if dist < 0.025:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # RIGHT CLICK (3 fingers approx)
            x3, y3 = lm[16].x, lm[16].y
            dist2 = math.hypot(x3 - x1, y3 - y1)

            if dist2 < 0.05:
                pyautogui.rightClick()
                time.sleep(0.5)

            # SCROLL
            if y < 0.3:
                pyautogui.scroll(20)
            elif y > 0.7:
                pyautogui.scroll(-20)

    cv2.imshow("AI Gesture Mouse PRO", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()