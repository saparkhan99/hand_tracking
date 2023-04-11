import mediapipe as mp
import skimage.io
import skimage
import skimage.filters
import cv2
import numpy as np
import math

from AR10_class import AR10
import time

robt_hand = AR10()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

def add_sp_noise(img, sp_ratio):
    gimg = skimage.util.random_noise(img, "s&p", amount = sp_ratio)
    return gimg

def add_gaussian_noise(img, var):
    gimg = skimage.util.random_noise(img, "gaussian", var = var)
    return gimg    

def nothing(x):
    pass

cv2.namedWindow("trackbars")
cv2.namedWindow("MediaPipe Hands")

#Trackbars for min and max thresholds for HSV components
cv2.createTrackbar('Salt and pepper noise', 'trackbars', 0, 100, nothing)
cv2.createTrackbar('Gaussian variance', 'trackbars', 0, 100, nothing)
cv2.createTrackbar('Add Gaussian?', 'trackbars', 0, 1, nothing)
cv2.createTrackbar('Add Median?', 'trackbars', 0, 1, nothing)
cv2.createTrackbar('Add Laplacian?', 'trackbars', 0, 1, nothing)
cv2.createTrackbar('Add sine noise?', 'trackbars', 0, 1, nothing)
cv2.createTrackbar('Noise wavelenght (x10)', 'trackbars', 5, 10, nothing)
cv2.createTrackbar('Add Fourier?', 'trackbars', 0, 1, nothing)



var = 0
sp_ratio = 0
median_filter = 0
gaus_filter = 0
lap = 0


joint_list = [[[1, 4, 3, 2, 0]], [[20,19,18,17,0]], [[16,15,14,13, 0]], [[12,11,10,9,0]], [[8,7,6,5,0]]]
memory = np.zeros((5, 2, 5))


def calculate_angle(p1, p2, p3):
    v1 = p3 - p2
    v2 = p1 - p2

    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    dist1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    dist2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

    angle = math.acos(dot / (dist1 * dist2))
    angle = np.abs(angle*180.0/np.pi)

    return angle


def draw_finger_angles(image, results, joint_list, index):
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 

        memory[index, :, :-1] = memory[index, :, 1:]

        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z]) # Third coord

            d = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z]) # Second coord
            e = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z]) # Third coord
            f = np.array([hand.landmark[joint[3]].x, hand.landmark[joint[3]].y, hand.landmark[joint[3]].z]) # Forth coord
            
            g = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z]) # Third coord
            h = np.array([hand.landmark[joint[3]].x, hand.landmark[joint[3]].y, hand.landmark[joint[3]].z]) # Forth coord
            i = np.array([hand.landmark[joint[4]].x, hand.landmark[joint[4]].y, hand.landmark[joint[4]].z]) # Wrist coord 
            
            
            angle_2 = calculate_angle(d, e, f)
            angle_3 = calculate_angle(g, h, i)
            
            if angle_2 > 180.0:
                angle_2 = 360-angle_2
            if angle_3 > 180.0:
                angle_3 = 360-angle_3

            
            memory[index, 0, -1] = angle_2
            memory[index, 1, -1] = angle_3

            mean_angle_2 = np.mean(memory[index, 0])
            mean_angle_3 = np.mean(memory[index, 1])

            robt_hand.set_angle(2 * index + 1, mean_angle_2)
            robt_hand.set_angle(2 * index, mean_angle_3 + 20)
                
            # cv2.putText(image, str(round(angle_1, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(mean_angle_2)), tuple(np.multiply(e[:2], [512, 512]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(mean_angle_3)), tuple(np.multiply(h[:2], [512, 512]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    

    return image

cap = cv2.VideoCapture(1)
    
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()

        height, width, _ = frame.shape

        frame = frame[:, (width-height)//2 : width-(width-height)//2, :]
        frame = cv2.resize(frame, (512, 512))

        # R, G, B = cv2.split(frame)

        # output1_R = cv2.equalizeHist(R)
        # output1_G = cv2.equalizeHist(G)
        # output1_B = cv2.equalizeHist(B)

        # frame = cv2.merge((output1_R, output1_G, output1_B))
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        
        # Flip on horizontal
        image = cv2.flip(image, 1)

        four = cv2.getTrackbarPos('Add sine noise?', 'trackbars')
        if four == 1:
            x = np.arange(0, 512, 1)
            y = np.arange(0, 512, 1)

            X, Y = np.meshgrid(y, x)

            wavelength = cv2.getTrackbarPos('Noise wavelenght (x10)', 'trackbars') * 10
            grating = 50*np.sin(2 * np.pi * X / wavelength) + 100

            ft = np.fft.fft2(grating)
            ft = np.fft.fftshift(ft)
            masked_values = np.argwhere(ft > 200000)

            noisy_image = np.zeros(image.shape, dtype=np.uint8)

            noisy_image[:,:,0] = image[:,:,0]*0.5 + grating * 0.5
            noisy_image[:,:,1] = image[:,:,1]*0.5 + grating * 0.5
            noisy_image[:,:,2] = image[:,:,2]*0.5 + grating * 0.5
            
            image = noisy_image.astype(np.uint8)

        

        if var == 0 :
            # CODE FOR SALT AND PEPPER NOISE
            sp_ratio = cv2.getTrackbarPos('Salt and pepper noise', 'trackbars')/100
            median_filter = cv2.getTrackbarPos('Add Median?', 'trackbars')
        else:
            sp_ratio = 0
            median_filter = 0
        
        
        image = add_sp_noise(image, sp_ratio)
        image = (image*255).astype(np.uint8)

        # CODE FOR GAUSSIAN
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        if sp_ratio == 0:
            var = cv2.getTrackbarPos('Gaussian variance', 'trackbars')/100
            gaus_filter = cv2.getTrackbarPos('Add Gaussian?', 'trackbars')
        else:
            var = 0
            gaus_filter = 0
        
        
        image = add_gaussian_noise(image, var)
        image = (image*255).astype(np.uint8)
        sigma = 2.0
        
        if median_filter == 1:
            filtered = cv2.medianBlur(image,3)
            results = hands.process(filtered)
        
        if gaus_filter == 1:
            filtered = skimage.filters.gaussian(image, sigma=(sigma, sigma), multichannel=True)
            
            filtered = (filtered*255).astype(np.uint8)
            results = hands.process(filtered)
            
            
            
        if gaus_filter == 0 and median_filter == 0:
            filtered = image
            results = hands.process(filtered)
            
            

        #CODE FOR LAPLACIAN
        
        lap = cv2.getTrackbarPos('Add Laplacian?', 'trackbars')
        
        if lap ==1:
            kernel = np.array([[0,-1,0], [-1,5,-1],[0,-1,0]])
            img = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
            image_sharp = cv2.filter2D(src=img, ddepth = -1, kernel=kernel) 
            image_sharp = cv2.cvtColor(image_sharp, cv2.COLOR_BGR2RGB)
            
            filtered = image_sharp
            results = hands.process(filtered)

        four_fil = cv2.getTrackbarPos('Add Fourier?', 'trackbars')
        if four_fil and four == 1:
            clear_image = np.zeros(noisy_image.shape)

            for i in range(3):
                fourier = np.fft.fftshift(np.fft.fft2(filtered[:, :, i]))
                # filtered = fourier
                mask = np.ones(fourier.shape)
                # mask[253:257, 244:248] = 0
                # mask[253:257, 264:268] = 0
                for coord in masked_values:
                    if coord[0] == 256 and coord[1] == 256:
                        continue
                    for k in range(-1, 2):
                        for j in range(-1, 2):
                            mask[coord[0]+k, coord[1]+j] = 0
            
                # mask[masked_values] = 0
                fourier = fourier * mask

                idft_shift = np.fft.ifftshift(fourier)  # Move the frequency domain from the middle to the upper left corner
                ifimg = np.fft.ifft2(idft_shift)  # Fourier library function call
                ifimg = np.real(ifimg)
                min = ifimg.min()
                max = ifimg.max()
                norm = (ifimg - min) / (max - min)
                clear_image[:, :, i] = norm
            
            filtered = (clear_image*255).astype(np.uint8)
            results = hands.process(filtered)

        
        # Set flag
        # image.flags.writeable = False
        
        # Detections
        # results = hands.process(image)
        
        # Set flag to true
        # image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
        # Detections
        # print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(filtered, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
                # Render left or right detection
#                 if get_label(num, hand, results):
#                     text, coord = get_label(num, hand, results)
#                     cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw angles to image from joint list
            filtered = draw_finger_angles(filtered, results, joint_list[0], 0)
            filtered = draw_finger_angles(filtered, results, joint_list[1], 1)
            filtered = draw_finger_angles(filtered, results, joint_list[2], 2)
            filtered = draw_finger_angles(filtered, results, joint_list[3], 3)
            filtered = draw_finger_angles(filtered, results, joint_list[4], 4)
            
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('MediaPipe Hands', filtered)
        cv2.imshow('trackbars', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()