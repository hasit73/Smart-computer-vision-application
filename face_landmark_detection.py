import cv2
# from numpy.core.fromnumeric import nonzero
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean

class FaceAnalyzer:
    """ This class responsible to detect face landmarks and mark them
        contains following features:-
        - detect eye blink
        - recognize face expression
        - check mouth is open or not
        
    """
    def __init__(self):
        self.__mp_draw = mp.solutions.drawing_utils
        self.__mp_face_mesh = mp.solutions.face_mesh
        self.__face_detector = self.__mp_face_mesh.FaceMesh(static_image_mode=False,min_detection_confidence = 0.6,min_tracking_confidence = 0.6)
        self.__face_mesh_marks = []
        self.face_points = None
        self.__blink_count = 0
        self.__blink_prev_status = 0
        self.__mouth_open = None
        self.left_eye_points = [[144,160],
                    [145,159],
                    [153,158],
                    [154,157],
                    [161,163]]
        self.right_eye_points = [ [381,384],
                            [380,385],
                            [374,386],
                            [373,387],
                            [388,390] ]

        self.__lips_upper_points = [76,183,42,41,38,12,268,271,272,407,306]
        self.__lips_lower_points = [76,96,89,179,86,15,316,403,319,325,306]

    def draw_lips_points(self):
        """ Use face landmark points and compute distance between upper lip and lower lip
            and check mouth is open or not
        """
        points = self.face_points
        ## compute distance between two point one point is of upper lip and second point is of lower lip
        d = euclidean(points[12],points[15])
        ## here 15 is threshold distance 
        if(d>15):
            self.__mouth_open = True
        else:
            self.__mouth_open =False

    @classmethod
    def find_angle(cls,v0,v1):
        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        return np.degrees(angle)


    def face_alignment_detection(self,image):
        left_eye_point = np.array(self.face_points[161])
        right_eye_point = np.array(self.face_points[388])
        third_point = np.array([right_eye_point[0],left_eye_point[1]])
        angel = FaceAnalyzer.find_angle(third_point-left_eye_point , right_eye_point-left_eye_point)
        rotated_status = None
        d = right_eye_point[1] - left_eye_point[1]
        if(d>0 and d>10):
            rotated_status = "right-rotated",str(round(angel,2))
        elif(d<0 and d<-10):
            rotated_status = "left-rotated",str(round(angel,2))
        else:
            rotated_status = "straight",str(round(angel,2))
        return rotated_status

        # cv2.line(image,left_eye_point,right_eye_point,(0,255,0),3)

    def eye_blink_detect(self):
        """ Use face landmark points to recognize eye blink event and 
            maintain eye blink counter
        """
        points = self.face_points
        
        left_dist = euclidean(points[153],points[158])
        right_dist = euclidean(points[374],points[386])
        if(left_dist<7 and right_dist<7):
            if(self.__blink_prev_status==0):
                self.__blink_count+=1
            self.__blink_prev_status=1
        else:
            self.__blink_prev_status = 0
            
    def face_emotion_detection(self):
        """
            Use face landmark points to identify user face expression
            some lips and eyes landmarks points were used to identify expression

        """
        points = self.face_points
        emotion = "None"

        if(self.__mouth_open):
            left_dist = euclidean(points[153],points[158])
            right_dist = euclidean(points[374],points[386])
            if(left_dist>9.5 and right_dist>9.5):
                emotion = "surprise"
        else:
            lips_upper_dist = 0
            lips_lower_dist= 0
            for i in range(len(self.__lips_upper_points)-1):
                lips_upper_dist+=euclidean( points[self.__lips_upper_points[i]],points[self.__lips_upper_points[i+1]])
                lips_lower_dist+=euclidean( points[self.__lips_lower_points[i]],points[self.__lips_lower_points[i+1]])
            lips_upper_dist = lips_upper_dist/(len(self.__lips_upper_points)-1)
            lips_lower_dist = lips_lower_dist/(len(self.__lips_lower_points)-1)   
            diff = lips_lower_dist-lips_upper_dist
            if(diff < -0.10):
                emotion = "sad"
            elif( -0.10 < diff < 0.015 ):
                emotion = "normal"
            else:
                emotion = "happy"
        return emotion

    def get_face_points(self,image):
        """ Get input image and find face landmark points and save them inside face_points instance variable"""
        self.face_points = []
        self.__face_mesh_marks = []
        results = self.__face_detector.process(image)
        ## Total 468 landmarks 
        if(results.multi_face_landmarks):
            for face in results.multi_face_landmarks:
                for lm in face.landmark:
                    x = lm.x
                    y = lm.y
                    h,w,c = image.shape
                    relative_x = int(x*w)
                    relative_y = int(y*h)
                    self.face_points.append([relative_x,relative_y])

                self.__face_mesh_marks.append(face)


    def process_frame(self,image,draw_mesh = False):
        """
            Take input image and manage all features of FaceAnalyzer
            call to all available features
        """
        self.get_face_points(image)
        cv2.rectangle(image,[20,30],[230,200],(255,255,255),-1)
        if(len(self.face_points)):
            self.eye_blink_detect()
            self.draw_lips_points()
            align_status,align_angle = self.face_alignment_detection(image)
            emotion = self.face_emotion_detection()
            cv2.putText(image,"Emotion : "+emotion,[30,50],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
            cv2.putText(image,"Alignment : "+align_status,[30,80],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
            cv2.putText(image,"Align Angle : "+align_angle,[30,110],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 

        cv2.putText(image,"Blink : "+str(self.__blink_count),[30,140],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
        cv2.putText(image,"Mouth Open : "+str(self.__mouth_open),[30,170],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
        if(draw_mesh):
            for face_lm in self.__face_mesh_marks:
                self.__mp_draw.draw_landmarks(image,face_lm,self.__mp_face_mesh.FACEMESH_CONTOURS )
        return image
