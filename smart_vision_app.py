from hand_landmark_detection import HandDetector
from face_landmark_detection import FaceAnalyzer
from virtual_drawingpad import VirtualDrawBoard
from filters import FilterApplyer
import cv2
import numpy as np
from time import time

class SmartVisionApp:

    """ Manage all operations
        provide interface to communicate between different modules like FilterApplyer, HandDetecor etc
    """
    def __init__(self):
        self.__hand_detector = HandDetector(min_detection_confidence=0.7,min_tracking_confidence=0.7)
        self.__face_analyzer = FaceAnalyzer()
        self.__virtual_drawer = VirtualDrawBoard(self.__hand_detector)
        self.__filter_applyer = FilterApplyer()
        self.__current_mode = "home"
        self.__available_home_modes = ["home","face","hands","draw"]
        self.__available_filters_modes = ["filters","hsv","hls","cartoon","gray","histeq","blurry"]

        self.__filters_options = {"home":[[550,20],[620,90]],
                                        "hsv":[[550,100],[620,150]],
                                        "hls":[[550,160],[620,210]],
                                        "gray":[[550,220],[620,270]],
                                        "cartoon":[[550,280],[620,330]],
                                        "histeq":[[550,340],[620,390]],
                                        "blurry":[[550,400],[620,450]],
                                        }
        self.__options_positions = {
                                        "home":[[550,20],[620,90]],
                                        "face":[[550,100],[620,170]],
                                        "hands":[[550,180],[620,250]],
                                        "draw":[[550,260],[620,330]],
                                        "filters":[[550,340],[620,430]]
                                    }
                                       
    def __draw_home_options(self,image):
        """ Draw different options (home,filter,hands,draw etc) on video frame"""
        options = {}        
        if(self.__current_mode in self.__available_filters_modes   ):
            options = self.__filters_options
        elif(self.__current_mode in self.__available_home_modes):
            options = self.__options_positions

        for option_name,pos in options.items():
            cv2.rectangle(image,pos[0],pos[1],(0,0,0),-1)
            cv2.putText(image,f"{option_name}",(pos[0][0]+10 , pos[0][1]+35 ),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 210), 1)


    def __controller(self,first_finger,second_finger):
        """
            Control different tasks 
            and check current operation mode

        """

        if(self.__current_mode in self.__available_filters_modes):
            options = self.__filters_options
        elif(self.__current_mode in self.__available_home_modes):
            try:
                cv2.destroyWindow("Filtered image")
            except:
                pass
            options = self.__options_positions

        for option_name,pos in options.items():
            if(VirtualDrawBoard.check_inside_rectangle(pos,first_finger) and VirtualDrawBoard.check_inside_rectangle(pos,second_finger)):
                self.__current_mode = option_name.lower()




    def process(self,image):
        """ 
            Perform dedicated operations based on selected operation mode
            example:
            if current operation mode = draw
            then perform operations of virtual drawing
        """
        points = self.__hand_detector.find_hands(image)
        if(len(points)):
            cv2.circle(image,points[8],2,(0,255,0),-1)
            cv2.circle(image,points[12],2,(0,255,255),-1)

            self.__controller(points[8],points[12])
            
            if(self.__current_mode=="hands"):
                self.__hand_detector.draw_fingers(image,points)
            elif(self.__current_mode == "draw"):
                self.__virtual_drawer.draw(image,points)
        if(self.__current_mode=="face"):
            self.__face_analyzer.process_frame(image,draw_mesh=False)
        elif(self.__current_mode in self.__available_filters_modes):
            if(self.__current_mode!="filters"):
                new_image , filter_name = self.__filter_applyer.apply_filter(image,filter_name=self.__current_mode)
                if(filter_name=="gray"):
                    new_image = cv2.cvtColor(new_image,cv2.COLOR_GRAY2BGR)
                cv2.rectangle(new_image,(20,20),(180,60),(0,0,0),-1)
                cv2.putText(new_image,f"Filter : {filter_name}",(20,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 250, 0), 2)
                cv2.imshow("Filtered image",new_image)
                cv2.waitKey(1)
        self.__draw_home_options(image)




### driver code to execute main SmartVisionApplication
if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    WINDOW_WIDTH , WINDOW_HEIGHT = 640,480
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cap.set(3,WINDOW_WIDTH)
    cap.set(4,WINDOW_HEIGHT)
    vision_app = SmartVisionApp()
    while True:
        ret,im = cap.read()
        im = cv2.flip(im,1)
        orig_im = im.copy()
        st = time()
        vision_app.process(im)
        et = time()
        fps = int(1/(et-st))
        cv2.putText(im,f"FPS : {fps}",(400,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("image",im)
        k = cv2.waitKey(10)
        if(k==27):
            break
