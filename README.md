# Smart-computer-vision-application

### Backend : opencv and python
### Library required:

- opencv = '4.5.4-dev'
- scipy = '1.4.1'
- numpy = '1.19.2'
- mediapipe = '0.8.9.1'


### NOTE:

- All required code files uploaded so don't need to install any external files or application

# Quick Overview about structure

#### 1) smart_vision_app.py

- Manage different operation modes and perform specific operations.
- Draw landmarks and display output image.


#### 2) virtual_drawingpad.py

- use hand_landmark_detection module to detect hands landmarks
- use finger point corrdinates to draw lines (set of points) 


#### 3) hand_landmark_detection.py

- use mediapipe lib to detect hand landmarks.
- count number of fingers


#### 4) face_landmark_detection.py

- use mediapipe lib to detect face landmarks.
- recognize facial expression from face key points
- maintain eye blink counter
- other operations


#### 5) filters.py

- apply different inbuilt filters(HSV, HLS, GRAY)
- apply differnet custom filters(cartoon,histeq,blurry)

# How to use 

1) clone this directory
 
2) use following command to run detection and tracking on your custom video

  ```
  python smart_vision_app.py
  ```
  
- Note : Before executing this command make sure that you have installed all required libs and all above .py files reside in same folder

### Results

- output:1 (operation mode : Draw)
![Draw mode](https://user-images.githubusercontent.com/69752829/148802457-cf18dc03-aa47-431a-8bb4-a55451782d05.mp4)

- output:2 (operation mode : Face )
![Face mode](https://user-images.githubusercontent.com/69752829/148802384-c4b7904d-3f7e-467e-a9c7-5e01656820d1.mp4)

- output:3 (operation mode : Filter)
![Filter mode](https://user-images.githubusercontent.com/69752829/148797658-6e32dc36-89ba-470e-b228-dd1032795bac.mp4)

- output:4 (operation mode : Hands )
![Hands mode](https://user-images.githubusercontent.com/69752829/148800380-2ee83f46-c860-42c4-8146-7409aea4e56a.mp4)

## If it's helpful for you then please give star Thank You :)
