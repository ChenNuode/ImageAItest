"""
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()


detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(person=True)

detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "pic3.jpg"), output_image_path=os.path.join(execution_path , "image2new.jpg"), minimum_percentage_probability=50)


for eachObject in detections:
	print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
	print("--------------------------------")
"""
from imageai.Detection import ObjectDetection
import os
#import cv2

execution_path = os.getcwd()


#camera = cv2.VideoCapture(0)

#detector = VideoObjectDetection()
detector = ObjectDetection()

detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(person=True)

#video_path = detector.detectCustomObjectsFromVideo(custom_objects=custom_objects, camera_input=camera,
                                #output_file_path=os.path.join(execution_path, "camera_detected_video")
                                #, frames_per_second=70, log_progress=True, minimum_percentage_probability=70)


#print(video_path)

returned_image, detections, extracted_objects = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image="pic2.jpg", output_type="array", extract_detected_objects=True, minimum_percentage_probability=70)

for things in extracted_objects:
    print(things)


