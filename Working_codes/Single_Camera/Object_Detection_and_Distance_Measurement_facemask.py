################################################################################################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import time
from threading import Thread
import pyzed.sl as sl
import cv2
import numpy as np
import math

def main():
    confThreshold= 0.5
    nmsThreshold = 0.4
    #Load YOLO
    modelConfiguration = '/home/nvidia/Project_codes/yolov3/cfg/yolov3-tiny.cfg'
    modelWeights = '/home/nvidia/Project_codes/yolov3/weights/yolov3-tiny.weights'

    faceMaskConfiguration = '/home/nvidia/Project_codes/yolov3/cfg/yolov3-tiny-face_mask.cfg'
    faceMaskModelWeights = '/home/nvidia/Project_codes/yolov3/weights/yolov3-tiny-face_mask.weights'

    #Read a network model stored in Darknet model files
    #Parameters: Configuration file, Weights file
    #Return: Net   
    net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
    facenet = cv2.dnn.readNetFromDarknet(faceMaskConfiguration, faceMaskModelWeights)

    #Set Backend and Target
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)



    #Create model from Deep learning network
    model = cv2.dnn_DetectionModel(net)
    modelFace = cv2.dnn_DetectionModel(facenet)

    #Store class name in a list
    LABELS = []
    LABELS_MASK = []

    with open('/home/nvidia/Project_codes/yolov3/data/coco.names', 'r') as f:
        LABELS = [line.strip() for line in f.readlines()]
    with open('/home/nvidia/Project_codes/yolov3/data/mask_classes.names', 'r') as f:
        LABELS_MASK = [line.strip() for line in f.readlines()]
    COLORS = [[0, 0, 255], [30, 255, 255], [0,255,0]]




    def detectObject_YOLO(modelName,img):
        modelName.setInputParams(size=(320, 320), scale=1/255, swapRB=True)
        image_test = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        image = image_test.copy()
       
        #print('image',image.shape)

        classes, confidences, boxes = modelName.detect(image, confThreshold, nmsThreshold)
        
        return classes,confidences,boxes
     
    frame_count = 0
  
    # Create a Camera object
    zed = sl.Camera()

	# Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    if not zed.is_opened():
        print("Opening ZED Camera...")
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit(1)

    #Set runtime parameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    #runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD

    image_zed = sl.Mat()
    depth_image_zed = sl.Mat();
    point_cloud = sl.Mat();

    #return zed,image_zed,point_cloud,runtime_parameters

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:

            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_image(depth_image_zed)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            
            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()

            classes,confidences,boxes = detectObject_YOLO(model,image_ocv)
            classes_face,confidences_face,boxes_face = detectObject_YOLO(modelFace,image_ocv)

            font = cv2.FONT_HERSHEY_SIMPLEX
            person_count = 0
            peopleWithMasks = 0
            peopleWithNoMasks = 0

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
            indexes_face = cv2.dnn.NMSBoxes(boxes_face, confidences_face, confThreshold, nmsThreshold)

            for i in range(len(boxes)):
                if i in indexes:
                    if str(LABELS[classes[i]]) == "person":
                        person_count = person_count + 1
                        
            for i in range(len(boxes_face)):
                if i in indexes_face:
                    if str(LABELS_MASK[classes_face[i]]) == "Mask":
                       peopleWithMasks = peopleWithMasks + 1
                elif str(LABELS_MASK[classes_face[i]]) == "No Mask":
                    peopleWithNoMasks = peopleWithNoMasks + 1



            #depth_image_ocv = depth_image_zed.get_data()


        
            for cl,score,(left,top,width,height) in zip(classes,confidences,boxes):
                start_point = (int(left),int(top))
                end_point = (int(left+width),int(top+height))
            
                x = int(left + width/2)
                y = int(top + height/2)

                color = COLORS[0]

                img =cv2.rectangle(image_ocv,start_point,end_point,color,3)
                cv2.putText(img, "No of People: " + str(person_count), (10, 25), font, 1, (160, 4, 183), 2)
                cv2.putText(img, "People with masks: " + str(peopleWithMasks) + ", People with no masks: " + str(peopleWithNoMasks), (10, 50), font, 1, (160, 4, 183), 2)
              # img = cv2.circle(img,(x,y),5,[0,0,255],5)
                text = f'{LABELS[cl]}'
                cv2.putText(img, text, (int(left), int(top-7)), cv2.FONT_ITALIC, 1, COLORS[0], 2 )
            
                x = round(x)
                y = round(y)

                err, point_cloud_value = point_cloud.get_value(x, y)
                distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] + point_cloud_value[1] * point_cloud_value[1] + point_cloud_value[2] * point_cloud_value[2])
                distance_text = "Distance: {:.2f}m".format(distance)
                cv2.putText(img, distance_text, (int(left), int(top + 25)), cv2.FONT_HERSHEY_COMPLEX, 1, COLORS[1], 2)
                cv2.imshow("Image",img)

            for cl_face,score_face,(left,top,width,height)in zip(classes_face,confidences_face,boxes_face):
                start_point = (int(left),int(top))
                end_point = (int(left+width),int(top+height))
            
                x = int(left + width/2)
                y = int(top + height/2)

                color = COLORS[cl_face]

                img =cv2.rectangle(image_ocv,start_point,end_point,color,3)
              # img = cv2.circle(img,(x,y),5,[0,0,255],5)
                text = f'{LABELS_MASK[cl_face]}'
                cv2.putText(img, text, (int(left), int(top-7)), cv2.FONT_ITALIC, 1, COLORS[0], 2 )
        

                frame_count = frame_count + 1
              #  i = i + 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    exit_flag = False


    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

"""
    while 1:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv2.imshow("ZED",image.get_data())
            key = cv2.waitKey(5)
    cv2.destroyAllWindows()

    # Close the camera
    zed.close()
    print("\nFINISH")
"""
if __name__ == "__main__":
    main()

