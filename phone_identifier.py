import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import os
from datetime import datetime

# Constants
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CROP_SIZE = 224
CHECK_INTERVAL = 5
SCREENSHOT_DIR = "screenshots"

class PhoneIdentifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📱 Phone Identifier")
        self.root.geometry("720x650")
        self.root.configure(bg="#1e1e1e")

        self.model = MobileNetV2(weights='imagenet')
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            self.root.quit()

        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

        self.frame_counter = 0
        self.last_prediction = "None"
        self.last_confidence = 0.0
        self.current_frame = None

        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        style = ttk.Style()
        style.configure("TLabel", foreground="white", background="#1e1e1e", font=("Segoe UI", 14))
        style.configure("TButton", font=("Segoe UI", 12), padding=6)

        self.canvas = tk.Canvas(self.root, width=FRAME_WIDTH, height=FRAME_HEIGHT, bg="#2c2c2c", highlightthickness=0)
        self.canvas.pack(pady=10)

        self.label = ttk.Label(self.root, text="Object: None", font=("Segoe UI", 14))
        self.label.pack(pady=(5, 0))

        self.label_conf = ttk.Label(self.root, text="Confidence: 0%", font=("Segoe UI", 12))
        self.label_conf.pack(pady=(0, 10))

        self.btn_record = ttk.Button(self.root, text="📸 Take Screenshot", command=self.save_screenshot)
        self.btn_record.pack(pady=(5, 5))

        self.btn_quit = ttk.Button(self.root, text="Quit", command=self.quit)
        self.btn_quit.pack(pady=5)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = frame_rgb.copy()

        if self.frame_counter % CHECK_INTERVAL == 0:
            self.last_prediction, self.last_confidence = self.classify_center_crop(frame_rgb)

        self.draw_feedback(frame_rgb, self.last_prediction, self.last_confidence)

        img = Image.fromarray(frame_rgb)
        img = img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.label.config(text=f"Object: {self.last_prediction}")
        self.label_conf.config(text=f"Confidence: {self.last_confidence:.1f}%")

        self.frame_counter += 1
        self.root.after(15, self.update_frame)

    def classify_center_crop(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        cx, cy = w // 2, h // 2
        x1, y1 = cx - CROP_SIZE // 2, cy - CROP_SIZE // 2
        crop = frame_rgb[y1:y1 + CROP_SIZE, x1:x1 + CROP_SIZE]

        img = preprocess_input(np.expand_dims(crop.astype(np.float32), axis=0))
        preds = self.model.predict(img, verbose=0)
        top_pred = decode_predictions(preds, top=1)[0][0]

        label = top_pred[1].replace("_", " ").capitalize()
        confidence = top_pred[2] * 100
        return label, confidence

    def draw_feedback(self, frame_rgb, label, confidence):
        h, w = frame_rgb.shape[:2]
        cx, cy = w // 2, h // 2
        x1, y1 = cx - CROP_SIZE // 2, cy - CROP_SIZE // 2
        x2, y2 = x1 + CROP_SIZE, y1 + CROP_SIZE
        color = (0, 255, 127) if confidence > 50 else (255, 80, 80)

        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label} ({confidence:.1f}%)"
        cv2.putText(frame_rgb, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def save_screenshot(self):
        if self.current_frame is not None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"screenshot_{now}.png")
            img_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img_bgr)
            print(f"Screenshot saved to {path}")

    def quit(self):
        self.cap.release()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = PhoneIdentifierApp(root)
    root.mainloop()
