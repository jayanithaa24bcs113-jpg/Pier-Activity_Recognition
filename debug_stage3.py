from ultralytics import YOLO

# Load Stage 3 model
model = YOLO("runs/detect/models/pier_cap_reinforcement/weights/best.pt")

# Replace this with one of your Stage 3 test images
image_path = "datasets/Pier_cap_reinforcement/test/images/test5.jpg"

results = model.predict(
    source=image_path,
    conf=0.15,
    device="cpu",   # we'll test CPU first
)

for r in results:
    print("\nModel's class names:")
    print(r.names)

    if r.boxes is None or len(r.boxes) == 0:
        print("\nNO DETECTIONS FOUND")
    else:
        print("\nDetections:")
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"class={cls} ({r.names[cls]})   confidence={conf:.3f}")