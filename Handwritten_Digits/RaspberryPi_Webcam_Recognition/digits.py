import cv2
import numpy as np
import skimage as ski
from skimage.restoration import _denoise
from skimage.filters import thresholding
import imutils
from imutils import object_detection
from tensorflow.keras.models import load_model

loaded_model = load_model("./cnn_model.h5")
n_pad = 0


def digitsDetector(image_color):
    
    image_color_s = imutils.resize(image_color, width=min(300, image_color.shape[1]))

    image_ = cv2.cvtColor(image_color_s, cv2.COLOR_BGR2GRAY)
    blurred = 255 * _denoise.denoise_tv_chambolle(image_, 1e-2)
    edged = cv2.Canny(blurred.astype(np.uint8), 25, 100)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    return filterBestCandidates(cnts, image_color_s, image_)


def padwithtens(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def mnistTransformation(image, angle_rotation):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = thresholding.threshold_otsu(image)
    image = 255 * np.where(image < thresh, (1 - image), 0).astype(np.uint8)
    (x, y, w, h) = cv2.boundingRect(image) 
    image = image[y : y + h, x: x + w]
    image = ski.util.pad(image, int(0.35 * max(image.shape)), padwithtens)
    if angle_rotation:
        image = ski.transform.rotate(image, angle_rotation)
    image = ski.transform.resize(image, (28, 28), anti_aliasing=True)
    #image = _denoise.denoise_tv_chambolle(image, 1e-2)
    image = 255 * image / image.max()
    return image


def fromPathMnistTrasformation(im_path, angle_rotation=None):
    image = ski.io.imread(im_path)
    return  mnistTransformation(image, angle_rotation)


def fromImageMnistTrasformation(image, angle_rotation=None):
    return  mnistTransformation(image, angle_rotation)  


def filterBestCandidates(cnts, image_color_s, image_):
    
    valid_boxes = []
    scores = []
    
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = image_[y - n_pad : y + h + n_pad, x - n_pad : x + w + n_pad]
        try:
            roi_tr = fromImageMnistTrasformation(roi)
            pred_score = loaded_model.predict(roi_tr.reshape(1, 28, 28, 1) / 255).max()
            valid_boxes.append((x - n_pad, y - n_pad, x + w + n_pad, y + h + n_pad))
            scores.append(pred_score)
            
        except:
            pass
        
    boxes = object_detection.non_max_suppression(np.array(valid_boxes), probs=np.array(scores), overlapThresh=0.01)
    
    for (startX, startY, endX, endY) in boxes:
        
        try:
            roi = image_[startY : endY, startX : endX]
            roi_tr = fromImageMnistTrasformation(roi)
            pred_digit = loaded_model.predict(roi_tr.reshape(1, 28, 28, 1) / 255).argmax()

            cv2.rectangle(image_color_s, (startX, startY), (endX, endY), (57, 255, 20), 2)
            cv2.putText(image_color_s, str(pred_digit), (endX, endY), 0, 1.0, (57, 255, 20), 4)
            
        except:
            pass
        
    return image_color_s


if __name__ == "__main__":

	a = 0
	video = cv2.VideoCapture(0)

	while True:
        a += 1
    	check = video.grab() #grab frame
        
        if a % 5 == 0:
            check, frame = video.retrieve()
    		print(check)
    		print(frame)
    		frame_with_boxes = digitsDetector(frame)

		cv2.imshow("Capturing", frame_with_boxes)
    	key = cv2.waitKey(1)
    
    	if key == ord('q'):
            break

	video.release()
	cv2.destroyAllWindows()
