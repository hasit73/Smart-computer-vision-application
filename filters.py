import cv2
import numpy as np

class FilterApplyer:
    
    """  This class Apply different filters on video frames
        Some default and specialized filters implemented in this class
        """
    def __init__(self):
        self.__default_filters = ["gray","hsv","hls"] 
        self.__special_filters = ["cartoon","blurry","histeq"]

    def apply_filter(self,image,filter_name="gray"):
        """ Takes 2 parameters:
            1) image :- input image - np.ndarray 
            2) filter_name :- Name of filter - str
            
            return 2 things :
            1) image :- output image after applying filter - np.ndarray
            2) filter_name :- Name of filter - str
            
        """
        if(filter_name in self.__default_filters):
            image = cv2.cvtColor(image,eval(f"cv2.COLOR_BGR2{filter_name.upper()}"))
            return image,filter_name
        elif(filter_name in self.__special_filters):
            if(filter_name=="cartoon"):
                image = self.make_cartoon(image)
                return image,filter_name
            elif(filter_name=="blurry"):
                image = cv2.medianBlur(image,7)
                return image,filter_name
            elif(filter_name=="histeq"):
                image = cv2.equalizeHist(self.apply_filter(image,"gray")[0])
                return cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),filter_name
        else:
            return image,"Normal"
    
    def make_cartoon(self,image):

        """ Takes 1 parameters:
            image :- input image - np.ndarray

            Returns 1 result:
            cartoon :- output image

        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_1 = cv2.medianBlur(gray, 9)
        edges = cv2.adaptiveThreshold(gray_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, d=9, sigmaColor=200,sigmaSpace=200)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

### driver code to test single module
# ## to ensure that Filter module works properly 

# if __name__=="__main__":

#     cap = cv2.VideoCapture(0)
#     WINDOW_WIDTH , WINDOW_HEIGHT = 640,480
#     cv2.namedWindow("image",cv2.WINDOW_NORMAL)
#     cap.set(3,WINDOW_WIDTH)
#     cap.set(3,WINDOW_HEIGHT)
#     f1 = FilterApplyer()
#     while True:
#         ret,im = cap.read()
#         im = cv2.flip(im,1)
#         new_im,fname = f1.apply_filter(im,"cartoon")
#         if(fname=="gray"):
#             new_im = cv2.cvtColor(new_im,cv2.COLOR_GRAY2BGR)
#         cv2.rectangle(new_im,(20,20),(180,60),(0,0,0),-1)
#         cv2.putText(new_im,f"Filter : {fname}",(20,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 250, 0), 2)
#         cv2.imshow("image",new_im)
#         k = cv2.waitKey(10)
#         if(k==27):
#             break