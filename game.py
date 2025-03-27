import cv2
import mediapipe as mp
import time
import win32gui
import win32con
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from pynput.keyboard import Controller, Key

# Initialize Keyboard Controller
keyboard = Controller()

# Open Hill Climb Racing Lite in Browser
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get("https://poki.com/en/g/hill-climb-racing-lite")

# Allow time for the game to load
time.sleep(10)

# Bring browser to front
def bring_browser_to_front():
    time.sleep(2)
    hwnd = win32gui.GetForegroundWindow()
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)

bring_browser_to_front()

# Initialize MediaPipe Hands & Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Sensitivity Thresholds
head_movement_threshold = 0.03  # Lower value for better responsiveness
prev_nose_x = None
is_accelerating = False  # Track acceleration state

def is_hand_compressed(hand_landmarks):
    """Check if the hand is compressed (fist gesture)"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y

    return index_tip > index_dip and middle_tip > middle_dip  # Fingers bent = Fist

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    frame = cv2.flip(frame, 1)  # Flip for natural movement
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    action = ""

    # Detect Hand Compression (Throttle "D" / Brake "A")
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            if is_hand_compressed(hand_landmarks):
                if not is_accelerating:  # Prevent repeated key presses
                    action = "Accelerating"
                    keyboard.press("d")  # Press and hold "D" for throttle
                    keyboard.release("a")  # Release "A" (Brake)
                    is_accelerating = True
            else:
                if is_accelerating:  # Prevent repeated key releases
                    action = "Braking"
                    keyboard.release("d")  # Release throttle
                    keyboard.press("a")  # Press and hold "A" for braking
                    is_accelerating = False

    # Detect Head Movement (Tilt Car)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            nose_x = face_landmarks.landmark[1].x  # Normalized nose position
            
            if prev_nose_x is not None:
                if nose_x - prev_nose_x > head_movement_threshold:
                    action = "Tilting Forward"
                    keyboard.press(Key.right)
                    time.sleep(0.1)
                    keyboard.release(Key.right)
                elif prev_nose_x - nose_x > head_movement_threshold:
                    action = "Tilting Backward"
                    keyboard.press(Key.left)
                    time.sleep(0.1)
                    keyboard.release(Key.left)

            prev_nose_x = nose_x

    # Display Action
    cv2.putText(frame, f"Action: {action}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show Webcam Feed
    cv2.imshow("Hill Climb Racing Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanudadadp
cap.release()
cv2.destroyAllWindows()
