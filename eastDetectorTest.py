from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract as pyt

pyt.pytesseract.tesseract_cmd = R'C:/Users/User/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'
file_name='C:\\Users\\User\\Desktop\\1.png'
east='C:/Users/User/Desktop/frozen_east_text_detection.pb'
min_confidence=0.5
width=320
height=320

# load the input image and grab the image dimensions
image=cv2.imread(file_name)
orig_image=image.copy()
text_extract_image=image.copy()
(H,W) = image.shape[:2]

# 새로운 width와 height를 설정하고 비율을 구한다
(newW,newH) = width,height
rW=W/float(newW)
rH=H/float(newH)

# image의 size를 재설정하고 새 이미지의 dimension을 구한다
image=cv2.resize(image,(newW,newH))
(H,W) = image.shape[:2]

layerNames=[
    'feature_fusion/Conv_7/Sigmoid',
    'feature_fusion/concat_3'
]

# load the pre-trained EAST text detector
print('[INFO] loading EAST text detector...')
net=cv2.dnn.readNet(east)

blob=cv2.dnn.blobFromImage(image,1.0,(H,W),
    (123.68,116.78,103.94),swapRB=True,crop=False)
start=time.time()
net.setInput(blob)

# geometry는 우리의 input image로 부터 bounding box좌표를 얻게해준다
# scores는 주어진 지역에 text가 있는지에 대한 확률을 준다
(scores,geometry) = net.forward(layerNames)
end=time.time()

print('[INFO] text detection took {:.6f} seconds'.format(end-start))

# scores의 크기를 받고 bounding box 사각형을 추출한뒤 confidencs scores에 대응해본다
(numRows,numCols) = scores.shape[2:4]
rects=[]
confidences=[]
for y in range(0,numRows):
    scoresData=scores[0,0,y]
    xData0=geometry[0,0,y]
    xData1=geometry[0,1,y]
    xData2=geometry[0,2,y]
    xData3=geometry[0,3,y]
    anglesData=geometry[0,4,y]

    for x in range(0,numCols):
        # 만약 score가 충분한 확률을 가지고 있지 않다면 무시한다
        if scoresData[x] < min_confidence:
            continue
        
        # 우리의 resulting feature map은 input_image보다 4배 작을것 이기 때문에
        # offset factor를 계산한다
        (offsetX,offsetY) = (x*4.0,y*4.0)

        # prediciton에 대한 회전각을 구하고 sin,cosine을 계산한다
        # 글씨가 회전되어 있을때를 대비
        angle=anglesData[x]
        cos=np.cos(angle)
        sin=np.sin(angle)

        # geometry volume를 사용해 bounding box의 width 와 height를 구한다
        h=xData0[x] + xData2[x]
        w=xData1[x] + xData3[x]

        # text prediction bounding box의 starting, ending (x,y) 좌표를 계산한다
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)
        
        # bounding box coordinates와 probability score를 append한다
        rects.append((startX,startY,endX,endY))
        confidences.append(scoresData[x])

# non-maxima suppression 을 weak,overlapping bounding boxes을 없애기위해 적용해준다
boxes=non_max_suppression(np.array(rects),probs=confidences)

def textRead(image):
    # apply Tesseract v4 to OCR 
    config = ("-l eng --oem 1 --psm 7")
    text = pyt.image_to_string(image, config=config)
    # display the text OCR'd by Tesseract
    print("OCR TEXT : {}\n".format(text))
    
    # strip out non-ASCII text 
    text = "".join([c if c.isalnum() else "" for c in text]).strip()
    print("Alpha numeric TEXT : {}\n".format(text))
    return text

for (startX,startY,endX,endY) in boxes:
    # 앞에서 구한 비율에 따라서 bounding box 좌표를 키워준다
    startX=int(startX * rW)
    startY=int(startY * rH)
    endX=int(endX * rW)
    endY=int(endY * rH)
    
    text=textRead(text_extract_image[startY:endY, startX:endX])

    cv2.rectangle(orig_image,(startX,startY),(endX,endY),(0,255,0),2)
    cv2.putText(orig_image, text, (startX, startY-10),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
  
print(text)
cv2.imshow('Text Detection', orig_image)
cv2.waitKey(0)