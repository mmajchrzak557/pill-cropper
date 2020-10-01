import numpy as np
import cv2
import imutils
import os
import time

def find_circle(image):
    img = image.copy()
    img = cv2.medianBlur(img, 5)
    mask = np.zeros(img.shape)
    try:
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1, 100,
                                    param1=100, param2=20, minRadius=0, maxRadius=50)       
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(mask,(i[0], i[1]), i[2], (255), -1)
    except:
        #print('No circles found')
        mask = np.ones(img.shape).astype('uint8')
    return mask.astype('uint8')

def crop(image, w, h):
    im_h, im_w = image.shape[:2]
    y1 = int(im_h/2) - int(h/2)
    y2 = int(im_h/2) + int(h/2)
    x1 = int(im_w/2) - int(w/2)
    x2 = int(im_w/2) + int(w/2)
    return image[y1:y2, x1:x2]

def get_rotation(mask, image, draw_rect=False):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    #if len(contours) > 1 : print('More than 1 object detected')
    if len(contours) == 0: return [w/2, h/2], 0, False
    c = contours[0]
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = [cX, cY]
    rotrect = cv2.minAreaRect(c)
    if rotrect[1][0] < 20 or rotrect[1][1] < 20:
        #print('error evaluating contour')
        is_circle = False
    else:
        is_circle = (rotrect[1][0]/rotrect[1][1] < 1.3 and rotrect[1][1]/rotrect[1][0] < 1.3)
        #string = 'circle' if is_circle else 'something else'
        #print(string)
        if draw_rect:
            box = cv2.boxPoints(rotrect)
            box = np.int0(box)
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    return center, rotrect[2], is_circle

def draw_contour(mask, image):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours) 
    #if len(contours) > 1 : print('More than 1 contour detected')
    if len(contours) == 0: return
    c = contours[0]
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 3, (0, 0, 255), -1)
    
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def improve_grabcut(image, new_mask, mask, b_model, f_model):
        new_mask[mask == 0] = cv2.GC_PR_BGD
        new_mask[mask == 255] = cv2.GC_PR_FGD
        new_mask, b_model, f_model = cv2.grabCut(image,new_mask,None,b_model,f_model,7,cv2.GC_INIT_WITH_MASK)
        new_mask = np.where((new_mask==2)|(new_mask==0),0,1).astype('uint8')
        original_image = image.copy()
        image = image*new_mask[:,:,np.newaxis]
        if not is_empty(image, 0.01):
            return image
        else:           
            return original_image    

def is_empty(image, percent):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    n = percent * image.shape[0]*image.shape[1]
    return cv2.countNonZero(image) < n
    
def get_filenames(path):
    f = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            f.append(filename)
    return f


start_time = time.time()

np.random.seed(88)

main_path = '%s\\demo' % (os.getcwd())
filenames = get_filenames(main_path)
#filenames = np.random.permutation(filenames)

##### INPUT PARAMETERS ####
w, h = 260, 260      # CROPPED IMAGE SIZE
w_d, h_d = 130, 130  # DISPLAY IMAGE SIZE
n = 18               # NUMBER OF INPUT IMAGES
gc_iter = 5          # GRABCUT ALGORITHM ITERATIONS
show_input = False   # SHOWING INPUT IMAGES
show_details = False # CONTOUR AND CENTER
row_length = 9       # NUMBER OF PICTURES IN ONE ROW
save = True         # SAVES IMAGES IN PATH\ZROBIONE IF TRUE
bw = False           # BLACK AND WHITE IMAGES IF TRUE

w_offset, h_offset = int(w/9), int(h/7)
out_height = 2*h_d if show_input else h_d
out_height *= (n-1)//row_length + 1
if not save: output = np.uint8(np.zeros((out_height, w_d*row_length, 3)))
for i in range(n):
    
    # LOAD, RESIZE AND CROP
    path = '%s\\%s' % (main_path, filenames[i])
    print(path)
    #print(str(round(i/n*100)) + '%')
    image = cv2.imread(path)
    #new_size = (int(image.shape[1]/8), int(image.shape[0]/8))
    #image = cv2.resize(image, new_size)
    image = crop(image, w, h)
    input_image = image.copy()
    original = image.copy()
       
    # ADAPTIVE THRESHOLD FOR IMPROVING GRABCUT
    ad_thresh = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ad_thresh = cv2.adaptiveThreshold(ad_thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,21,5)
    kernel = np.ones((5, 5), np.uint8)   
    t_mask = cv2.erode(ad_thresh, kernel, iterations = 2)
    t_mask = cv2.dilate(t_mask, kernel, iterations = 1)
    
    # GRABCUT
    gc_mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    rect = (w_offset, h_offset, w - 2*w_offset, h - 2*h_offset)
    cv2.grabCut(image.copy(), gc_mask, rect, bgdModel, fgdModel, gc_iter, cv2.GC_INIT_WITH_RECT)
    gc_mask2 = np.where((gc_mask == 2)|(gc_mask == 0), 0 ,1).astype('uint8')
    image = image*gc_mask2[:, :, np.newaxis]
    if cv2.countNonZero(t_mask) > 0 and 'Miflonide_Breezhaler' not in filenames[i%2]:
        image = improve_grabcut(image, gc_mask, t_mask, bgdModel, fgdModel)
        
    # THRESHOLD, EROSION AND DILATION
    mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    image = cv2.bitwise_and(image, image, mask = mask)
    c, a, is_circle = get_rotation(mask, input_image, True)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # DRAW GRABCUT RECTANGLE ON INPUT IMAGE
    if show_input and show_details:
        cv2.rectangle(input_image, (w_offset, h_offset),
                      (w - w_offset, h - h_offset), [0, 0, 255], thickness = 2)
        draw_contour(mask.copy(), input_image)
           
    # HOUGH CIRCLES AND MASK
    changed = False
    if is_circle:
        circ_mask = find_circle(image)
        np.uint8(image)
        if not is_empty(image, 0.025):
            image = cv2.bitwise_and(image, image, mask = circ_mask)
            changed = True
    
    # GET COLOR IMAGE
    if not bw:
        final_mask = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)[1]    
        image = cv2.bitwise_and(original, original, mask = final_mask)
    
    # ROTATE AND SHIFT TO CENTER + GREYSCALE
    mask = circ_mask if changed else mask
    center, angle, ic = get_rotation(mask, input_image)
    x = int(image.shape[1]/2) - center[0]
    y = int(image.shape[0]/2) - center[1]
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])   
    image = cv2.warpAffine(image, translation_matrix, (w, h))
    image = rotate_image(image, angle)
    
    # CROP AND RESIZE
    #image = crop(image, 130, 130)
    #image = cv2.resize(image, (28, 28))
    #print(image.shape)
    #image = cv2.resize(image, (w, h))
    
    # ADD TO OUTPUT
    if not save:
        if bw: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if show_input:
            out = cv2.resize(np.concatenate((input_image, image), axis = 0), (w_d, 2*h_d))
            height = h_d*2
        else:
            out = cv2.resize(image, (h_d, w_d))
            height = h_d
        x, y = int(i % row_length)*w_d, int(i//row_length)*height
        output[y : y + height, x : x + w_d, :] = out
    else:
        #folder = filenames[i%2][:-8]
        #save_path = '%s\\zrobione\\%s\\%s' % (os.getcwd(), folder, filenames[i])
        save_path = '%s\\sm\\%s' % (os.getcwd(), filenames[i%2])
        cv2.imwrite(save_path, image)

time = round((time.time() - start_time), 2)
print("\nExecuted in %s seconds" %time)

if not save:
    cv2.imshow('OUTPUT', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows