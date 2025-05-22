# 🖐️ Hand Tracking and Gesture Recognition with MediaPipe

This project uses **OpenCV** and **MediaPipe** to detect hands in real time from a webcam feed, measure the distance between the thumb and index finger of the right hand, and classify the state ("Open" or "Closed") of the left hand.

---

## 📸 Features

- Detects up to **two hands** simultaneously
- **Classifies hand** as Left or Right using MediaPipe metadata
- **Measures distance** between thumb and index finger on the right hand
- **Classifies open/closed gesture** on the left hand by comparing fingertip positions
- Draws hand landmarks and labels on the live video stream

---

## 🛠️ Requirements

Install dependencies using pip:

```bash
pip install opencv-python mediapipe numpy
