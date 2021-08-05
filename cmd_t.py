import cv2
wTitle = 'jpg'
# cv2.namedWindow('jpg', cv2.WINDOW_NORMAL)
# c = cv2.imread('E:\\odessa food market.jpg')
# height, width = c.shape[:2]
# c1 = cv2.resize(c, (int(width/4), int(height/4)))
# #cv2.resizeWindow('jpg', width, height)
# cv2.imshow('jpg', c1)
# r = cv2.waitKey(0)
# print( "DEBUG: waitKey returned:", chr(r))
# cv2.destroyAllWindows()

import imutils

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    frame = imutils.resize(frame, width=320)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()