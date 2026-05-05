import cv2
import os
from ultralytics import YOLO

# #importing pretrain model
# cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_fullbody.xml"
# personCM = cv2.CascadeClassifier(cascPath)      # set up pretrain model on faces

# #import car detection
# carCascPath = os.path.dirname(__file__)+"/data/cars.xml"
# carCM = cv2.CascadeClassifier(carCascPath)

# #import fire detection
# fireCascPath = os.path.dirname(__file__)+"/data/fire.xml"
# fireCM = cv2.CascadeClassifier(fireCascPath)

# # func to display detected object
# # vid = input vector of pics
# def display(vid, cascadeModel,  w_treshold = 0,minsize = (30,30), color = (0, 255, 0)):
#     objs = cascadeModel.detectMultiScale3(
#         vid, 
#         scaleFactor=1.1, 
#         minNeighbors=5, 
#         minSize=(30,30), 
#         outputRejectLevels=True,
#     )

#     # Draw a rectangle around the faces
#     for (weight, (x, y, w, h)) in zip(objs[2], objs[0]):
#         if (weight > w_treshold):
#             cv2.rectangle(frames, (x, y), (x+w, y+h), color, 2)
#             cv2.putText(frames, f"{weight:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)



# YOLO person & car detection
detectModel = YOLO("yolov8n.pt")

#car detect function
def detectCar(vid):
    results = detectModel(vid, stream=False, classes=[2])    # classes 2 = car

    for r in results:
        boxes=r.boxes
        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # confidence
            cls = int(box.cls[0])      # class id

            # Draw bounding box
            cv2.rectangle(vid, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label with class and confidence
            label = f"{detectModel.names[cls]} {conf:.2f}"
            cv2.putText(vid, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


#person detect function
def detectPerson(vid):
    results = detectModel(vid, stream=False, classes=[0])    # classes 0 = person

    for r in results:
        boxes=r.boxes
        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # confidence
            cls = int(box.cls[0])      # class id

            # Draw bounding box
            cv2.rectangle(vid, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put label with class and confidence
            label = f"{detectModel.names[cls]} {conf:.2f}"
            cv2.putText(vid, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            


# YOLO fire & smoke detection
fireDetectModel = YOLO(os.path.dirname(__file__)+"/models/yolov8fire_detection.pt")

#fire & smoke detect function
def detectFire(vid):
    results = fireDetectModel(vid, stream=False)    # classes 0 = personfire, class 1 = smoke

    for r in results:
        boxes=r.boxes
        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # confidence
            cls = int(box.cls[0])      # class id

            # Draw bounding box
            cv2.rectangle(vid, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Put label with class and confidence
            label = f"{fireDetectModel.names[cls]} {conf:.2f}"
            cv2.putText(vid, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            


# YOLO crashed car door detection
crashedCarDoorModel = YOLO(os.path.dirname(__file__)+"/models/crashedCarDoor.pt")

# car door
def detectCrashedDoor(vid):
    results = crashedCarDoorModel(vid, stream=False)    # classes 0 = personfire, class 1 = smoke

    for r in results:
        boxes=r.boxes
        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])  # confidence
            cls = int(box.cls[0])      # class id

            # Draw bounding box
            cv2.rectangle(vid, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Put label with class and confidence
            label = f"{crashedCarDoorModel.names[cls]} {conf:.2f}"
            cv2.putText(vid, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# export models

# function to put detections on picture
def detectAndLabel(img_path:str, output_dir:str = None):

    if not output_dir:
        output_dir = os.path.dirname(img_path)

    # get input img
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    
    # process input img
    detectCar(img)
    detectFire(img)
    detectPerson(img)
    detectCrashedDoor(img)

    # get into new file name 
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}_new{ext}"

    new_image_path = os.path.join(output_dir, new_name)

    # write img
    cv2.imwrite(new_image_path, img)

    return new_image_path


# # vid to use
# video = os.path.dirname(__file__)+"/test_vid/accident1.avi"
# video_capture = cv2.VideoCapture(video)   # input source

# # display
# while video_capture.isOpened():
#     # Capture frame-by-frame
#     ret, frames = video_capture.read()

#     gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)   #convert color

#     # no grey scale needed for YOLOv8
#     detectPerson(frames)
#     detectCar(frames)
#     detectFire(frames)

#     # Display the resulting frame
#     cv2.imshow('Video', frames)      # window name

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# video_capture.release()
# cv2.destroyAllWindows()