import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt

file1 = "IMG_8253.JPG"

def images(file):
  
    path = os.path.join(os.path.dirname(__file__), "dataset_for_coins")
    image = os.path.join(path, file)
    img = cv2.imread(image)
    img_resized = cv2.resize(img, (1024, 1280))
    img_resized = cv2.medianBlur(img_resized, 7)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # GaussianBlur is used to remove gaussian noise from the image.
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    contours = cv2.Canny(gray_blur, 30, 135)
    thresh = cv2.adaptiveThreshold(contours, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 1)

    kernel = np.ones((1, 1), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=10)

    closing_ = closing.copy()
    contours, _ = cv2.findContours(closing_, cv2.RETR_EXTERNAL, \
                                               cv2.CHAIN_APPROX_SIMPLE)

    for contour in range(len(contours)):
        cv2.drawContours(img_resized, contours[contour], -1, (0, 0, 255),5)
        x,y,w,h = cv2.boundingRect(contours[contour])
        cv2.putText(img_resized, 'Coin-' + str(contour+1), (x-10,y-10), cv2.FONT_ITALIC,1.1,(120,200,0),3)

    showImage(img_resized)

def showImage(img):

    plt.figure(figsize=(16,9))
    plt.title("Enclosed contours")
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def run_main():
    # os.path.join(os.path.dirname(__file__), "Dataset_Coins")
    cap = cv2.VideoCapture('bb.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while(True):
        ret, frame = cap.read()
        # roi = frame[0:500, 0:500]
        roi = frame[0:720, 320:1000]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # GaussianBlurring is used to remove gaussian noise from the image.
        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        # cv.AdaptiveThreshold(src, dst, maxValue, adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C, thresholdType=CV_THRESH_BINARY, blockSize=3, param1=5) → ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C
        thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
        # kernel = np.ones((3, 3), np.uint8)
        kernel = np.ones((1, 1), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=10)
        cont_img = closing.copy()
        contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, \
                                               cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000 or area > 4000:
                continue
            if len(cnt) < 5:
                continue
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(roi, ellipse, (0,255,0), 2)
        cv2.imshow("Morphological Closing", closing)
        cv2.imshow("Adaptive Thresholding", thresh)
        cv2.imshow('Contours', roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # run_main()
    images(file = file1)