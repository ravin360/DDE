import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression, scale_boxes

# Load YOLOv5 model
model_path = "C:/Users/sravi/Downloads/yolov5m.pt"  # Path to pre-trained YOLOv5 model
image_size = 640  # Image size for YOLO input

def load_yolo_model():
    device = select_device('cpu')  # Select device (CPU or GPU)
    model = DetectMultiBackend(model_path, device=device, dnn=False)
    return model, device

def detect_objects_yolo(model, device, image):
    img = cv2.resize(image, (image_size, image_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)

    # Run YOLO model
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0  # Normalize
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.1, 0.3, max_det=1000)

    return pred

# Load the image
image_path = "C:/Users/sravi/Downloads/WhatsApp Image 2024-11-04 at 20.40.06_31d365c3.jpg"
image = cv2.imread(image_path)
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_equalized = cv2.equalizeHist(img_gray)
image_enhanced = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2BGR)

model, device = load_yolo_model()
print("Model Loaded")
pred = detect_objects_yolo(model, device, image_enhanced)

for det in pred:  # detections per image
    if det is not None and len(det):
        det[:, :4] = scale_boxes(image.shape[:2], det[:, :4], image.shape).round()
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"Class {int(cls)}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            top_edge = [(x1, y1), (x2, y1)]
            cv2.line(image, top_edge[0], top_edge[1], (255, 0, 0), 2)  # Blue top edge line
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_ITALIC, 0.5, (0, 255, 0), 2)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 7))
plt.imshow(image_rgb)
plt.title('Detected Object with Top Edge')
plt.axis('off')
plt.savefig("output1.png")
print("Plot saved as 'output1.png'")
