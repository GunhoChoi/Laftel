import cv2
i = 0

video_name='Onepunch3.mp4'

cap = cv2.VideoCapture('./videodata/'+video_name)

length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)

while(cap.isOpened()):
    ret, frame = cap.read()
    if i % fps == 0:
        cv2.imwrite("./videodata_frame/video_name+str(i) + ".jpg", frame)
        print(i)
    i += 1
    if i == length:
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
