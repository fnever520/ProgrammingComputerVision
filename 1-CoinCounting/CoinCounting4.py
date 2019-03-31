import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

file4 = "IMG_8256.JPG"

def images(file):

    count = 1 
    path = os.path.join(os.path.dirname(__file__), "dataset_for_coins")
    image = os.path.join(path, file)
    img = cv2.imread(image)
    img_resized = cv2.resize(img, (1024, 1280))
    img_resized = cv2.medianBlur(img_resized, 5)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # GaussianBlur is used to remove gaussian noise from the image.
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    contours = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1, 70, \
    	param1 = 53, param2 = 36, minRadius = 0, maxRadius =0 )
    # thresh = cv2.adaptiveThreshold(contours, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 1)

    contours = np.uint16(np.around(contours))

    for contour in contours[0,:]:
        
        cv2.circle(img_resized, (contour[0], contour[1]), contour[2], \
        	(0, 0, 255),5)
        cv2.circle(img_resized, (contour[0], contour[1]), 2, (0,255,0), 1)
        cv2.putText(img_resized, 'Coin-' + str(count), \
        	(contour[0]-80,contour[1]-70), cv2.FONT_ITALIC,1.1,(120,200,0),3)
        count +=1

    showImage(img_resized)

def showImage(img):

    plt.figure(figsize=(16,9))
    plt.title("Enclosed contours")
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == '__main__':
	images(file4)