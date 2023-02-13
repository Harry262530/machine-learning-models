import cv2

#img=cv2.imread("lena.png")
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
namesofClass= []
Class='coco.names'
with open(Class,'rt') as f:
    namesofClass=f.read().rstrip('\n').split('\n')
print(namesofClass)

configuration='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights='frozen_inference_graph.pb'

net=cv2.dnn_DetectionModel(weights,configuration)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
while True:
    success,img=cap.read()
    ClassIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(ClassIds, bbox)
    if len(ClassIds) != 0:
        for x, y, z in zip(ClassIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, z, color=(0, 255, 0), thickness=2)
            cv2.putText(img, namesofClass[x - 1].upper(), (z[0] + 10, z[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 255, 0),
                        2)


    cv2.imshow("Output", img)
    cv2.waitKey(1)




