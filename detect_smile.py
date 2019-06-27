from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2

cascadePath     = "haarcascade_frontalface_default.xml"
preTrainedModel = "lenet.hdf5"

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(cascadePath)
model = load_model(preTrainedModel)

video = cv2.VideoCapture(0)
while True:
    isGrabbed, frame = video.read()

    # resize the frame, convert it to grayscale
    frame = imutils.resize(frame, width=300)
    frameClone = frame.copy()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the input frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                        minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # loop over the face bounding boxes
    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image
        roi = gray[fY:fY + fH, fX:fX + fW]

        # resize ROI to a fixed 28x28 pixels
        roi = cv2.resize(roi, (28, 28))

        # prepare the ROI for classification via the CNN
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine the probabilities of both "smiling" and "not
        # smiling", then set the label accordingly
        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not Smiling"

        # display the label and bounding box rectangle on the output
        # frame
        if label == "Smiling":
            color = (0, 255, 0) # green boundary box
        else:
            color = (0, 0, 255) # red boundary box
        
        cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), color, 2)
    
    # show our detected faces along with smiling/not smiling labels
    cv2.imshow("Face", frameClone)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()