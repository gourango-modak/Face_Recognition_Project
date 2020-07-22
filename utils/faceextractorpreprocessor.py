import argparse
import cv2
import imutils
import os
from imutils.paths import list_images


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="haar cascade xml file path")
ap.add_argument("-d", "--datasets", required=True, help="image datasets folder")
ap.add_argument("-o", "--output", required=True, help="output directory")
arg = vars(ap.parse_args())


imageList = list_images(arg["datasets"])

dect = cv2.CascadeClassifier(arg["cascade"])


for i,im in enumerate(imageList):
    image = cv2.imread(im)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rect = dect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE)
    if not os.path.exists(os.path.sep.join([arg["output"], im.split(os.path.sep)[-2]])):
        os.mkdir(os.path.sep.join([arg["output"], im.split(os.path.sep)[-2]]))
    for (j,(fx,fy,fw,fh)) in enumerate(rect):
        cp = image[fy:fy+fh, fx:fx+fw]
        cv2.imwrite("{}.png".format(os.path.sep.join([arg["output"], im.split(os.path.sep)[-2], str(i+j).zfill(8)])), cp)