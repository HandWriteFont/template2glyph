# from email import iterators
import numpy as np
# from tkinter import Misc
from PIL import Image, ImageOps

import cv2
import numpy as np
# import scipy.misc as misc 


#데이터 전처리 함수(crop → resize → padding)

def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized

def tight_crop_image(img, verbose=False, resize_fix=False):
    img_size = img.shape[0]
    full_white = img_size
    col_sum = np.where(full_white - np.sum(img, axis=0) > 1) # axis가 0이면 열 단위의 합, 1이면 행 단위의 합
    row_sum = np.where(full_white - np.sum(img, axis=1) > 1) 
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    cropped_image = img[y1:y2, x1:x2]
    cropped_image_size = cropped_image.shape


    if verbose:
        print('(left x1, top y1):', (x1, y1))
        print('(right x2, bottom y2):', (x2, y2))
        print('cropped_image size:', cropped_image_size)
        
    if type(resize_fix) == int:
        origin_h, origin_w = cropped_image.shape
        if origin_h > origin_w:
            resize_w = int(origin_w * (resize_fix / origin_h))
            resize_h = resize_fix
        else:
            resize_h = int(origin_h * (resize_fix / origin_w))
            resize_w = resize_fix
        if verbose:
            print('resize_h:', resize_h)
            print('resize_w:', resize_w, \
                  '[origin_w %d / origin_h %d * target_h]' % (origin_w, origin_h))
        
        # resize
        cropped_image = cropped_image.resize(resize_h, resize_w)
        cropped_image = normalize_image(cropped_image)
        cropped_image_size = cropped_image.shape
        if verbose:
            print('resized_image size:', cropped_image_size)
        
    elif type(resize_fix) == float:
        origin_h, origin_w = cropped_image.shape
        resize_h, resize_w = int(origin_h * resize_fix), int(origin_w * resize_fix)
        if resize_h > 120:
            resize_h = 120
            resize_w = int(resize_w * 120 / resize_h)
        if resize_w > 120:
            resize_w = 120
            resize_h = int(resize_h * 120 / resize_w)
        if verbose:
            print('resize_h:', resize_h)
            print('resize_w:', resize_w)
        
        # resize
        cropped_image = cropped_image.resize(resize_h, resize_w)
        cropped_image = normalize_image(cropped_image)
        cropped_image_size = cropped_image.shape
        if verbose:
            print('resized_image size:', cropped_image_size)
    
    return cropped_image


def add_padding(img, image_size=128, verbose=False, pad_value=None):
    height, width = img.shape
    if not pad_value:
        pad_value = img[0][0]
    if verbose:
        print('original cropped image size:', img.shape)
    
    # Adding padding of x axis - left, right
    pad_x_width = (image_size - width) // 2
    pad_x = np.full((height, pad_x_width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_x, img), axis=1)
    img = np.concatenate((img, pad_x), axis=1)
    
    width = img.shape[1]

    # Adding padding of y axis - top, bottom
    pad_y_height = (image_size - height) // 2
    pad_y = np.full((pad_y_height, width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_y, img), axis=0)
    img = np.concatenate((img, pad_y), axis=0)
    
    # Match to original image size
    width = img.shape[1]
    if img.shape[0] % 2:
        pad = np.full((1, width), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=0)
    height = img.shape[0]
    if img.shape[1] % 2:
        pad = np.full((height, 1), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=1)

    if verbose:
        print('final image size:', img.shape)
    
    return img


def centering_image(img, image_size=128, verbose=False, resize_fix=False, pad_value=None):
    if not pad_value:
        pad_value = img[0][0]
    cropped_image = tight_crop_image(img, verbose=verbose, resize_fix=resize_fix)
    centered_image = add_padding(cropped_image, image_size=image_size, verbose=verbose, pad_value=pad_value)

    cv2.imshow("Cropped_tight",cropped_image)
    cv2.waitKey(0)

    return centered_image

# 후처리 함수
def morph(img):
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)

    kernel = np.ones((3,3), np.uint8)


    #closingimg = cv2.dilate(closingimg, kernel, iterations = 1)
    closingimg = cv2.dilate(img,kernel,iterations=1)
    closingimg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    ksize = 3
    closingimg = cv2.GaussinanBlur(closingimg, (ksize,ksize), 0)
    #closingimg = cv2.adaptiveThreshold(closingimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,49,5)
    #closingimg = cv2.morphologyEx(closingimg, cv2.MORPH_OPEN, kernel, iterations=2)
    #closingimg = cv2.adaptiveThreshold(closingimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,127,5)

    closingimg = cv2.resize(closingimg, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    return closingimg


def main():
    load_path = "./"
    save_path = "./"

    img = Image.open('Test.jpg')
    imgGray = img.convert('L')
    img_resize = imgGray.resize((1280, 777))
    x = 66
    y = 110
    letterN = 1
    words = ['값', '같', '곬', '곶', '깎', '넋', '늪', '닫', '닭', '닻', '됩', '뗌', '략', '몃', '밟', '볘', '뺐',
            '뽈', '솩', '쐐', '앉', '않', '얘', '얾', '엌', '옳', '읊', '죡', '쮜', '춰', '츄', '퀭', '틔', '핀', '핥', '훟']
    while letterN <= 36:
        croppedImage = img_resize.crop((x, y, x + 120, y + 120))
        croppedImage = croppedImage.resize((128, 128))
        pillow2array = np.array(croppedImage)
        th2 = cv2.adaptiveThreshold(pillow2array,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,127,5)
        th3 = cv2.adaptiveThreshold(pillow2array,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,127,5)
        centalcroppedImage = centering_image(th3, verbose=True)
        # croppedImage.save(save_path + '/' + words[letterN - 1] + '.jpg')

        x = x + 126

        if letterN == 9:
            x = 66
            y = 276
        elif letterN == 18:
            x = 66
            y = 444
        elif letterN == 27:
            x = 66
            y = 611

        letterN = letterN + 1
    # while letterN <= 36:
    #     croppedImage = img_resize.crop((x, y, x + 120, y + 120))
    #     croppedImage = croppedImage.resize((128, 128))
    #     pillow2array = np.array(croppedImage)
    #     ret,th = cv2.threshold(pillow2array,127,255, cv2.THRESH_BINARY)
    #     cv2.imshow(str(letterN),th)
    #     centalcroppedImage = centering_image(th, verbose=True)
    #     # croppedImage.save(save_path + '/' + words[letterN - 1] + '.jpg')

    #     x = x + 125

    #     if letterN == 9:
    #         x = 66
    #         y = 367
    #     elif letterN == 18:
    #         x = 66
    #         y = 590
    #     elif letterN == 27:
    #         x = 66
    #         y = 815

    #     letterN = letterN + 1
    

if __name__ == '__main__':
    main()

# def main():
#     load_path = "./"
#     save_path = "./"

#     # img = Image.open(load_path + '/sample2.jpg')
#     img = Image.open('./Test.jpg')
#     imgGray = img.convert('L')
#     img_resize = imgGray.resize((1713, 1063))
#     # template = cv2.imread('./Test.jpg', cv2.IMREAD_COLOR)
#     # cv2.imshow('Template',template)
#     # template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
#     # template = template.resize(1713, 1063)
#     x = 105
#     y = 143
#     letterN = 1
#     words = ['값', '같', '곬', '곶', '깎', '넋', '늪', '닫', '닭', '닻', '됩', '뗌', '략', '몃', '밟', '볘', '뺐',
#             '뽈', '솩', '쐐', '앉', '않', '얘', '얾', '엌', '옳', '읊', '죡', '쮜', '춰', '츄', '퀭', '틔', '핀', '핥', '훟']
#     while letterN <= 36:
#         croppedImage = img_resize.crop((x, y, x + 160, y + 160))
#         croppedImage = croppedImage.resize((128, 128))
#         pillow2array = np.array(croppedImage)
#         cv2.imshow(str(letterN),pillow2array)
#         centalcroppedImage = centering_image(pillow2array, verbose=True)
#         # croppedImage.save(save_path + '/' + words[letterN - 1] + '.jpg')

#         x = x + 168

#         if letterN == 9:
#             x = 105
#             y = 367
#         elif letterN == 18:
#             x = 105
#             y = 590
#         elif letterN == 27:
#             x = 105
#             y = 815

#         letterN = letterN + 1
# img = Image.open(load_path + '/sample2.jpg')
# imgGray = img.convert('L')
# img_resize = imgGray.resize((1713, 1063))

# x = 105; y = 143;
# letterN = 1;
# words = ['값','같','곬','곶','깎','넋','늪','닫','닭','닻','됩','뗌','략','몃','밟','볘','뺐','뽈','솩','쐐','앉','않','얘','얾','엌','옳','읊','죡','쮜','춰','츄','퀭','틔','핀','핥','훟']
# while letterN <= 36:
#   croppedImage = img_resize.crop((x, y, x + 160, y + 160))
#   croppedImage = croppedImage.resize((128, 128))
#   croppedImage.save(save_path + '/' + words[letterN - 1] + '.jpg')

#   x = x + 168

#   if letterN == 9:
#     x = 105; y = 367
#   elif letterN == 18:
#     x = 105; y = 590
#   elif letterN == 27:
#     x = 105; y = 815;

#   letterN = letterN + 1


# timg = cv2.imread(load_path + '/ch.png', cv2.IMREAD_GRAYSCALE)

# cv2.imwrite(load_path + '/ch2.png', morph(timg))

# t1img = cv2.imread(load_path + '/test.jpg', cv2.IMREAD_GRAYSCALE)
# new_img = centering_image(t1img)
# cv2.imwrite(load_path + '/test2.jpg', new_img)
