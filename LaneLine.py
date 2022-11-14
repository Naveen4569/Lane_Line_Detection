import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def make_coordinates(image,line_parameters):
    try:
        slope,intercept=line_parameters
    except TypeError:
        slope,intercept=0.001,0
    y1=image.shape[0]
    y2=int(y1*(3/5))

    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),10)
    return line_image

def weighted_image(image,a,line_image,b,c):
    return cv2.addWeighted(image,a,line_image,b,c)

def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.10, rows]
    top_left     = [cols*0.30, rows*0.6]
    bottom_right = [cols*0.95, rows]
    top_right    = [cols*0.55, rows*0.6]

    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver

def region_of_intrest(image,vertices):
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    mask=np.zeros_like(image)
    cv2.fillPoly(mask,vertices,ignore_mask_color)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < -0.4:
                left_fit.append((slope, intercept))
            elif slope>0.4:
                right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    print(left_fit_average, 'left')
    print(right_fit_average, 'right')
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    return cv2.HoughLinesP(img,rho,theta,threshold,np.array([]),minLineLength=min_line_len,maxLineGap=max_line_gap)


def gray_scale_image(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

def guassian_blur(img,k_size):
    return cv2.GaussianBlur(img,(k_size,k_size),0)

def canny(image,low_threshold,high_threshold):

    canny=cv2.Canny(image,low_threshold,high_threshold)
    return canny

def finding_lane(img):
    gray=gray_scale_image(img)

    blur_img=guassian_blur(gray,k_size=5)

    canny_img=canny(blur_img,low_threshold=28,high_threshold=84)

    ver=get_vertices(img)

    cropped_image=region_of_intrest(canny_img,vertices=ver)

    lines=hough_lines(cropped_image,rho=5,theta=np.pi/180,threshold=100,min_line_len=40,max_line_gap=100)

    average_lines=average_slope_intercept(img,lines)

    line_img=display_lines(img,average_lines)

    combo_image=weighted_image(img,a=0.6,line_image=line_img,b=1,c=1)

    return combo_image

if __name__ == '__main__':

    cap=cv2.VideoCapture('Sample_Videos/solidYellowLeft.mp4')
    while(cap.isOpened()):
        res,frame=cap.read()
        if res:
            final_img=finding_lane(frame)
            cv2.imshow('result',final_img)
            if cv2.waitKey(30) & 0xFF==ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
