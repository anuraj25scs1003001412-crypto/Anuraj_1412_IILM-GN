# 🧠 Face Recognition System

A real-time face recognition system built with Python and OpenCV using the LBPH (Local Binary Patterns Histograms) algorithm. It detects and identifies faces through a webcam feed.

---

## 📁 Project Structure

```
face-recognition/
├── colllect-faces.py     # Step 1: Capture face images from webcam
├── train_model.py        # Step 2: Train the LBPH model on collected images
├── recognize.py          # Step 3: Run real-time face recognition
├── final_demo.py         # All-in-one demo script
├── trainer.yml           # Saved trained model (generated after training)
├── labels.txt            # Label-to-name mapping (generated after training)
├── haarcascade_frontalface_default.xml  # Haar cascade for face detection
└── data/                 # Auto-created folder for collected face images
    └── <person_name>/
        └── *.jpg
```

---

## ⚙️ Requirements

- Python 3.x
- OpenCV with contrib modules

Install dependencies:

```bash
pip install opencv-python opencv-contrib-python numpy
```

---

## 🚀 How to Use

### Step 1 — Collect Face Images

Run the collection script and enter the person's name when prompted. It will capture **20 face images** via your webcam and save them to the `data/` folder.

```bash
python colllect-faces.py
```

- Press **Q** to quit early.
- Images are saved as grayscale face crops to `data/<name>/`.

---

### Step 2 — Train the Model

Once faces are collected, train the LBPH recognizer:

```bash
python train_model.py
```

This will:
- Read all images from the `data/` directory
- Detect faces using the Haar cascade
- Train the LBPH model
- Save the model to `trainer.yml`
- Save label mappings to `labels.txt`

---

### Step 3 — Run Face Recognition

Launch the real-time recognition:

```bash
python recognize.py
```

Or run the demo version:

```bash
python final_demo.py
```

- A **green box** with the person's name = recognized face (confidence < 80)
- A **red box** with "Unknown" = unrecognized face
- Press **Q** to quit

---

## 🔍 How It Works

1. **Face Detection** — Haar Cascade Classifier detects face regions in each video frame.
2. **Feature Extraction** — The detected face ROI (Region of Interest) is passed to the LBPH recognizer.
3. **Prediction** — The model predicts a label ID and a confidence score.
4. **Thresholding** — If confidence is below 80, the face is labeled with the known name; otherwise it's marked "Unknown".

---

## 🧩 Adding More People

Simply re-run `colllect-faces.py` with a new name, then re-run `train_model.py`. The model supports multiple people — each gets their own folder under `data/`.

---

## 📌 Notes

- `trainer.yml` and `labels.txt` are auto-generated — do not edit them manually.
- The `haarcascade_frontalface_default.xml` file is bundled with OpenCV and referenced via `cv2.data.haarcascades`, so you don't need to keep it locally unless required.
- Confidence scores in LBPH are distances — **lower is better**.

---

## 📄 License

This project is open source and free to use for educational purposes.
