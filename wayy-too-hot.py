import cv2
from deepface import DeepFace

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

# continuity camera is usually first, so try to skip it
cam = cv2.VideoCapture(1)
if not cam.read()[0]:
    cam = cv2.VideoCapture(0)

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"
genderList = ['Male', 'Female']
genderNet = cv2.dnn.readNet(genderModel, genderProto)
genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

while(True):
    ret, frame = cam.read()

    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bb in bboxes:
        padding = 200
        face80 = frame[max(0,bb[1]-padding):min(bb[3]+padding,frame.shape[0]-1),max(0,bb[0]-padding):min(bb[2]+padding, frame.shape[1]-1)]
        emotion = ""
        try:
            # we assume there's only one
            firstface = DeepFace.analyze(face80, actions=['emotion'])[0]
            output = firstface['emotion']
            emotion = max(output, key=output.get)
        except ValueError as e:
            print(e)

        padding = 20
        face20 = frame[max(0,bb[1]-padding):min(bb[3]+padding,frame.shape[0]-1),max(0,bb[0]-padding):min(bb[2]+padding, frame.shape[1]-1)]
        # opencv is BGR not RGB, so they have this handy option that we want off
        blob = cv2.dnn.blobFromImage(face20, 1.0, (227, 227), swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        if gender == 'Female':
            emotion = "Wayy too hot"

        label = f"{emotion}"
        cv2.putText(frameFace, label, (bb[0], bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frameFace)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

