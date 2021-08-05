# import the opencv library
import cv2
import numpy as np

# define a video capture object
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

# import cv2
# #import cv2.cv as cv
# from time import time
# boxes = []
# def on_mouse(event, x, y, flags, params):
#    # global img
#    t = time()
#
#    if event == cv2.EVENT_LBUTTONDOWN:
#       print('Start Mouse Position: '+str(x)+', '+str(y))
#       sbox = [x, y]
#       boxes.append(sbox)
#       # print count
#       # print sbox
#
#    elif event == cv2.EVENT_LBUTTONUP:
#       print('End Mouse Position: '+str(x)+', '+str(y))
#       ebox = [x, y]
#       boxes.append(ebox)
#       print(boxes)
#       crop = img[boxes[-2][1]:boxes[-1][1],boxes[-2][0]:boxes[-1][0]]
#
#       cv2.imshow('crop',crop)
#       k =  cv2.waitKey(0)
#       if ord('r')== k:
#          cv2.imwrite('Crop'+str(t)+'.jpg',crop)
#          print("Written to file")
#
# count = 0
# while(1):
#    count += 1
#    img = cv2.imread("E:\\odessa food market.jpg")
#    # img = cv2.blur(img, (3,3))
#    img = cv2.resize(img, None, fx = 0.25,fy = 0.25)
#
#    cv2.namedWindow('real image')
#    cv2.setMouseCallback('real image', on_mouse)
#    cv2.imshow('real image', img)
#    if count < 50:
#        if cv2.waitKey(33) == 27:
#            cv2.destroyAllWindows()
#            break
#        elif count >= 50:
#           if cv2.waitKey(0) == 27:
#              cv2.destroyAllWindows()
#              break
#           count = 0
