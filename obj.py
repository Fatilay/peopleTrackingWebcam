from ultralytics import YOLO
import cv2
import cvzone
import math
import time

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

prev_frame_time = 0
new_frame_time = 0

cap = cv2.VideoCapture(0)  # 0 webcam için 
cap.set(3, 640)
cap.set(4, 480)

#cap = cv2.VideoCapture("insan.mp4")

model = YOLO("yolov8n.pt")##l versiyonu daha iyi ama gpu kullanılırsa 

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_AREA)

    results = model(img, stream=True, classes=[classNames.index("person")])#img üzerinde nesneyi tanıma

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box  
            x1, y1, x2, y2 = box.xyxy[0]#resimde nesne algılandığında nesnenin içine alındığı dikdörtgenin koordinatları 
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))#dikdörtgen içinde göster cvzone özelliği
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100 #güven değeri
            # Class Name
            cls = int(box.cls[0])#nesne isimlerini int tanımla
            #tanımlanan nesnenin ismini yazma
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1) 

            cx, cy = x1 + w // 2, y1 + h // 2     #algılanan nesnenin merkez noktası
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            cx2, cy2 = 1280 // 2, 720 // 2
            cv2.circle(img, (cx2, cy2), 5, (255, 0, 255), cv2.FILLED)

            cv2.rectangle(img, (120, 120), (1280-120,720-120), (255, 0, 0), 2)

            cv2.line(img, (cx2,cy2), (cx,cy), (255, 0, 0), 1) #hedef noktaya uzaklık

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("fps: ", fps)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

