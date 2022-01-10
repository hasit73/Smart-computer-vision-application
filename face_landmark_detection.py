import cv2
import mediapipe as mp
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

    def eye_blink_detect(self):
        """ Use face landmark points to recognize eye blink event and 
            maintain eye blink counter
        """
        points = self.face_points
        
        left_dist = euclidean(points[153],points[158])
        right_dist = euclidean(points[374],points[386])
        if(left_dist<8 and right_dist<8):
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
        cv2.rectangle(image,[30,30],[220,170],(255,255,255),-1)
        if(len(self.face_points)):
            self.eye_blink_detect()
            self.draw_lips_points()
            emotion = self.face_emotion_detection()
            cv2.putText(image,"Emotion : "+emotion,[50,50],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
        cv2.putText(image,"Blink : "+str(self.__blink_count),[50,100],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
        cv2.putText(image,"Mouth Open : "+str(self.__mouth_open),[50,150],cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 
        if(draw_mesh):
            for face_lm in self.__face_mesh_marks:
                self.__mp_draw.draw_landmarks(image,face_lm,self.__mp_face_mesh.FACEMESH_CONTOURS )
        return image

### driver code to test single module
### to ensure that FaceAnalyzer module works properly 
# if __name__=="__main__":
#     cap = cv2.VideoCapture(0)
#     cap.set(3,640)
#     cap.set(4,480)
#     f1 = FaceAnalyzer()
#     while True:
#         ret,im = cap.read()
#         f1.process_frame(im)
#         cv2.imshow("Image",im)
#         k = cv2.waitKey(10)
#         if(k==27):
#             break
#         elif(k==32):
#             cv2.waitKey(-1)