import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open webcam
cap = cv2.VideoCapture(0)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# Labels (adjust based on your model's output)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'O', 13: 'P',
    14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'U',19:'V',
    20: 'W', 21: 'X', 22: 'Y', 23: 'Z', 24: '1', 25: '2', 26: '3', 27: '4', 28: '5',
    29: '6', 30: '7', 31: '8', 32: '9', 33: '10', 34: 'Namaste',
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape

    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    if results.multi_hand_landmarks:
        hand_data = []

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            single_hand = []
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
                single_hand.append((lm.x, lm.y))
            hand_data.append(single_hand)

        # Process up to 2 hands
        for i in range(2):
            if i < len(hand_data):
                hand = hand_data[i]
                x_base = min([pt[0] for pt in hand])
                y_base = min([pt[1] for pt in hand])
                for x, y in hand:
                    data_aux.append(x - x_base)
                    data_aux.append(y - y_base)
            else:
                # Pad with zeros for missing hand
                data_aux.extend([0] * 42)

        if len(data_aux) == 84:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), '?')

            # Draw prediction box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
