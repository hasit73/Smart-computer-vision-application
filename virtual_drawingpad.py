import cv2
# from hand_landmark_detection import HandDetector
from time import time
import numpy as np
from scipy.spatial.distance import euclidean

class VirtualDrawBoard:
    """
        VirtualDrawBoard module responsible to manage all tasks of virtual drawing
        - draw lines of points
        - check user point exists inside drawing box boundry
    """
    def __init__(self,hand_detector):
        self.hand_detector = hand_detector
        self.__draw_area = [[20,50],[300,250]]
        self.__clear_button = [[20,320],[120,400]]
        self.__history = [[]]
        self.__drawing_flag = 0
        self.__prev_drawing_flag = 0
        self.__Text = "Off"

    

    @classmethod
    def check_inside_rectangle(cls,rect,p2):
        """ check single point present inside one rectangle """
        p2 = np.array(p2)
        p1 = rect[0]
        p3 = rect[1]

        if(p1[0] < p2[0] and p2[0] < p3[0] and p1[1] < p2[1] and p2[1] < p3[1]):
            return True
        else:
            return False

    def draw(self,image,points):
        if(len(points)):
            tap_point = points[8]
            option_point = points[12]
            self.__prev_drawing_flag = self.__drawing_flag
            self.__drawing_flag = 0
            d = euclidean(option_point,tap_point)
            if(d < 30):
                self.__drawing_flag = 1
                self.__Text  = "ON"
            status = VirtualDrawBoard.check_inside_rectangle(self.__draw_area,tap_point)
            if(int(status) and self.__drawing_flag):
                if(self.__prev_drawing_flag):
                    self.__history[-1].append(tap_point)
                else:
                    self.__history.append([tap_point])
            if(VirtualDrawBoard.check_inside_rectangle(self.__clear_button,option_point) and VirtualDrawBoard.check_inside_rectangle(self.__clear_button,tap_point)):
                self.__history = [[]]

            cv2.rectangle(image,[10,410],[150,450],(255,255,255),-1)
            cv2.putText(image,f"Drawing : {self.__Text}",(20,430),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if self.__drawing_flag else (0,0,255), 2) 
        
        
        
        if(len(self.__history)>1): 
            for hist_points in self.__history:
                cv2.polylines(image,[np.array(hist_points).reshape(-1,1,2)],False,(255,255,0),2)
            
        cv2.rectangle(image,self.__draw_area[0],self.__draw_area[1],(255, 0, 0),2)
        cv2.putText(image,f"Drawing Area",(self.__draw_area[0][0]+20,self.__draw_area[0][1]-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.rectangle(image,self.__clear_button[0],self.__clear_button[1],(0,55,0),2)
        cv2.putText(image,f"Clear Button",(self.__clear_button[0][0]+10,self.__clear_button[0][1]-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,55,0), 1)


### driver code to test single module
### to ensure that VirtualDrawpad module works properly 

# if __name__=="__main__":
#     cap = cv2.VideoCapture(0)
#     WINDOW_WIDTH , WINDOW_HEIGHT = 640,480
#     cv2.namedWindow("image",cv2.WINDOW_NORMAL)
#     cap.set(3,WINDOW_WIDTH)
#     cap.set(4,WINDOW_HEIGHT)
#     hand_detector = HandDetector(min_detection_confidence=0.7,min_tracking_confidence=0.7)
#     v1 = VirtualDrawBoard(hand_detector)
#     while True:
#         ret,im = cap.read()
#         im = cv2.flip(im,1)
#         st = time()
#         points = v1.hand_detector.find_hands(im)
#         v1.draw(im,points)
#         et = time()
#         fps = int(1/(et-st))
#         cv2.putText(im,f"FPS : {fps}",(500,420),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#         cv2.imshow("image",im)
#         k = cv2.waitKey(10)
#         if(k==27):
#             break