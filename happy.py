import cv2
import time
import sys
import os
import signal
import subprocess
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

def main():
    # continuity camera is usually first, so try to skip it
    cam = cv2.VideoCapture(1)
    if not cam.read()[0]:
        cam = cv2.VideoCapture(0)

    faceProto = "models/opencv_face_detector.pbtxt"
    faceModel = "models/opencv_face_detector_uint8.pb"
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    # MAIN LOOP
    process = None
    last_happy = time.time() - 1
    was_happy = False
    is_happy = False
    for emotion in detection(cam, faceNet):
        if emotion == 'happy':
            last_happy = time.time()
        is_happy = (time.time() - last_happy) < 1

        if was_happy and not is_happy:
            os.kill(process.pid, signal.SIGTERM)
        if (not was_happy) and is_happy:
            process = subprocess.Popen(['afplay', 'media/queen.mp3'])

        was_happy = is_happy
    # END MAIN LOOP

    cam.release()
    cv2.destroyAllWindows()

# turn off pesky loading messages
def analyze(*args, **kwargs):
    f = open(os.devnull, 'w')
    sys.stderr = f
    ret = None
    try:
        ret = DeepFace.analyze(*args, **kwargs)
    except ValueError as e:
        print(e)
    finally:
        sys.stderr = sys.__stderr__
        return ret

def detection(cam, faceNet):
    # takes picture
    # caluclates boxes
    # yields emotion
    # shows picture
    while True:
        _, frame = cam.read()

        frameFace, bboxes = getFaceBox(faceNet, frame)
        emotion = ""
        for bb in bboxes:
            padding = 200
            face80 = frame[max(0,bb[1]-padding):min(bb[3]+padding,frame.shape[0]-1),max(0,bb[0]-padding):min(bb[2]+padding, frame.shape[1]-1)]
            faces = analyze(face80, actions=['emotion'])
            if faces:
                # we assume there's only one face detected by emotion network, 
                # since we are only showing it 1 bounding box
                output = faces[0]['emotion']
                emotion = max(output, key=output.get)
                yield emotion

            label = f"{emotion}"
            cv2.putText(frameFace, label, (bb[0], bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frameFace)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# # ideas:
# change the tempo based on the emotion
# you progress through the song whenever you're happy
main()
