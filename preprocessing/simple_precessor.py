import cv2



class SimplePrecessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store given width and height
        self.width = width
        self.height = height
        self.inter = inter
    
    def preprocess(self, image):
        # resize image to our own width and height
        return cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)