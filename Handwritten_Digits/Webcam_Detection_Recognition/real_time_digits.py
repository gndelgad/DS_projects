import numpy as np
import skimage as ski
import cv2
from imutils import object_detection, resize,grab_contours

import tensorflow as tf
from tensorflow.keras.models import load_model


def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def mnistTransformation(image):
    """
    Binarize image, pad, resize and denoise to be compatible to MNIST dataset.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = ski.filters.threshold_otsu(image)
    image = 255 * np.where(image < thresh, (1 - image), 0).astype(np.uint8)
    (x, y, w, h) = cv2.boundingRect(image) 
    image = image[y : y + h, x: x + w]
    image = ski.util.pad(image, int(0.35 * max(image.shape)), padwithtens)
    image = ski.transform.resize(image, (28, 28), anti_aliasing=True)
    image = ski.restoration.denoise_tv_chambolle(image, 1e-2)
    image = 255 * image / image.max()
    return image

def digitsRecognizer(image_color, origin_roi, digit_recognizer):
    """
    Detect edges and recognize digits.
    """
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(image_gray.astype(np.uint8), 100, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# contours
    cnts = grab_contours(cnts)
    
    valid_boxes, scores = [], []
    n_pad = 0
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = image_gray[y - n_pad : y + h + n_pad, x - n_pad : x + w + n_pad]
        try:
            roi_transformed = mnistTransformation(roi)
            pred_score = digit_recognizer.predict(roi_transformed.reshape(1, 28, 28, 1) / 255).max()
            valid_boxes.append((x - n_pad, y - n_pad, x + w + n_pad, y + h + n_pad))
            scores.append(pred_score)
        except:
            pass
        
    filtered_boxes = object_detection.non_max_suppression(np.array(valid_boxes), probs=np.array(scores), overlapThresh=0.01)
    predicted_digits = []
    startX0, startY0 = origin_roi
    
    for startX, startY, endX, endY in filtered_boxes:
        try:
            roi = image_gray[startY : endY, startX : endX]
            roi_transformed = mnistTransformation(roi)
            digit = digit_recognizer.predict(roi_transformed.reshape(1, 28, 28, 1) / 255).argmax()
            predicted_digits.append(((startX0 + startX, startY0 + startY, startX0 + endX, startY0 + endY), digit))
        except:
            pass
    
    return predicted_digits

def displayDigitsBoxes(image, predicted_digits):
    """
    Display image with detected object boxes and predicted digits.
    """
    for coordinates_box, digit in predicted_digits:
        startX, startY, endX, endY = coordinates_box
        cv2.rectangle(image, (startX, startY), (endX, endY), (57, 255, 20), 2)
        cv2.putText(image, str(digit), (endX, endY), 0, 0.7, (57, 255, 20), 1)
        
    return image

def DigitDetectorRecognizerInit():
    """
    Initialize EAST text detector and digit recognizer nets.
    """
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")
    
    text_detector = {'net': cv2.dnn.readNet('./frozen_east_text_detection.pb'),
                     'output': outputLayers}
    
    digit_recognizer = load_model("./cnn_model.h5")
    
    return text_detector, digit_recognizer

def textDetectionBoxes(image, text_detector):
    """
    Predict text boxes within an input image.
    """
    inpHeight, inpWidth = image.shape[:2]
    image_res = cv2.resize(image, (320, 320))
    blob = cv2.dnn.blobFromImage(image_res, 1.0, (320, 320), (123.68, 116.78, 103.94), True, False)
    
    text_detector['net'].setInput(blob)
    scores, geometry = text_detector['net'].forward(text_detector['output'])
 
    # Grab the number of rows and columns from the scores volume, then initialize our set of
    # bounding box rectangles and corresponding confidence scores
    (numRows, numCols) = scores.shape[2:4]
    min_confidence = 0.5
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # Extract the scores (probabilities), followed by the geometrical data used to derive
        # potential bounding box coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0, xData1, xData2, xData3 = geometry[0, 0 : 4, y]
        anglesData = geometry[0, 4, y]

    # Loop over the number of columns
        for x in range(0, numCols):
            # If our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue
            # Compute the offset factor as our resulting feature maps will be 4x smaller than input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # Extract the rotation angle for the prediction and then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # Use the geometry volume to derive the width and height of the bounding box
            h, w = xData0[x] + xData2[x], xData1[x] + xData3[x]
            # Compute both the starting and ending (x, y)-coordinates for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX, startY = int(endX - w), int(endY - h)
            # Add the bounding box coordinates and probability score to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
            
    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    boxes = object_detection.non_max_suppression(np.array(rects), probs=confidences, overlapThresh=0.3)

    rW, rH = inpWidth / 320, inpHeight / 320

    image_with_text_boxes = image.copy()
    scaled_boxes = []
    # Loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # Scale the bounding box coordinates based on the respective ratios
        startX, startY, endX, endY = int(startX * rW), int(startY * rH), int(endX * rW), int(endY * rH)
        # Draw the green bounding box on the image
        cv2.rectangle(image_with_text_boxes, (startX, startY), (endX, endY), (57, 255, 20), 2)
        scaled_boxes.append((startX, startY, endX, endY))
        
    return image_with_text_boxes, scaled_boxes

if __name__ == "__main__":
    
    text_detector, digit_recognizer = DigitDetectorRecognizerInit()
    video = cv2.VideoCapture(0)

    while True:

        check, frame = video.read()
        print(check)
        print(frame)

        image_with_text_boxes, text_boxes = textDetectionBoxes(frame, text_detector)

        predicted_digits = []
        min_area = 50

        for text_box in text_boxes:

            startX, startY, endX, endY = text_box
            text_roi = frame[startY : endY, startX : endX] # ROI = region of interest
            origin_roi = [startX, startY]
            hight, width, channels = text_roi.shape
            if hight * width < min_area:
                continue
            predicted_digits += digitsRecognizer(text_roi, origin_roi, digit_recognizer)

        cv2.imshow("Capturing", displayDigitsBoxes(frame, predicted_digits))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()