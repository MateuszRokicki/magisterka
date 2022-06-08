import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
#import numpy as np
import cv2
# import pandas as pd
# from glob import glob
# from tqdm import tqdm
# import tensorflow as tf
# from tensorflow.keras.utils import CustomObjectScope
# from metrics import dice_loss, dice_coef, iou
import matplotlib.pyplot as plt
import pixellib
from pixellib.instance import instance_segmentation
from pixellib.torchbackend.instance import instanceSegmentation
#from pixellib.tune_bg import alter_bg
from tune_bg import alter_bg
import imquality.brisque as brisque
import skimage

sys.path.append('../')
#import SRN.run_model
from SRN.run_model import main as SRNmain


def coordinates(fg_img):
    coord = []
    fg_img_copy = fg_img.copy()
    def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            coord.append(x)
            coord.append(y)
            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, str(x) + ',' +
            #             str(y), (x, y), font,
            #             1, (255, 0, 0), 2)
            #text = 'If choosen object is correct, click ESC. If You want to choose again, click any button'
            if len(coord) == 4:
                cv2.rectangle(fg_img, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 0), 2)
                text = ''
                cv2.putText(fg_img, text, (0, 100), font, 1, (0, 0, 255), thickness=3)
                cv2.imshow('Image', fg_img)
            else:
                cv2.imshow('Image', fg_img)



    # wait for a key to be pressed to exit
    t = True

    while t:
        cv2.imshow('Image', fg_img)
        cv2.setMouseCallback('Image', click_event)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            img = fg_img_copy
            cv2.destroyAllWindows()
            t = False
        else:
            coord = []
            img = fg_img_copy
            cv2.imshow('Image', img)
            cv2.setMouseCallback('Image', click_event)

    # close the window
    print('Coordinates')
    print(coord)
    return coord

def change_background(bg_img, fg_img, coord):
    img_crop = fg_img[coord[1]:coord[3], coord[0]:coord[2]]
    b_img_crop = bg_img[coord[1]:coord[3], coord[0]:coord[2]]
    # cv2.imshow('', img_crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    change_bg = alter_bg(model_type="pb")
    change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
    result_image = change_bg.change_bg_img(img_crop, b_img_crop)
    # cv2.imshow('',result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    bg_img[coord[1]:coord[3], coord[0]:coord[2]] = result_image
    cv2.imshow('',bg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('248.jpg', bg_img)
    print('Saved')
    return bg_img

def brisque_custom(img):
    brsq = brisque.score(img)
    return (brsq-100)/-1

def main():
    #path = sys.argv[2]
    #b_img = cv2.imread(path)
    # cv2.imshow('', b_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    b_img = cv2.imread(
          "D:/Studia/Magisterskie/Magisterka/programy/Scale-recurrent Network for Deep Image Deblurring/SRN-Deblur/248_blur.png")
    #db_info = sys.argv[1]
    # if db_info == '0' or db_info == 'SRN':
    #     print('SRN')
    #     db_img = SRNmain(path)
    # elif db_info == '1' or db_info == 'RLF':
    #     print('RLF')
    #     #db_img = SRNmain(path)
    #     pass
    # elif db_info == '2' or db_info == 'EBKS':
    #     print('EBKS')
    #     #db_img = SRNmain(path)
    #     pass
    # else:
    #     print('Wrong name')
    #     return
    #print(db_img)
    # db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('', db_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    db_img = cv2.imread(
         "D:/Studia/Magisterskie/Magisterka/programy/Scale-recurrent Network for Deep Image Deblurring/SRN-Deblur/248_deblur.png")
    # b_img + deblur => img
    coord = coordinates(db_img)
    final_img = change_background(b_img, db_img, coord)
    print(brisque_custom(b_img), brisque_custom(db_img), brisque_custom(final_img))
    cv2.imshow('', b_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('', final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #if(coord[1] < coord[3])


    # segment_image = instance_segmentation()
    # segment_image.load_model('mask_rcnn_coco.h5')
    #
    # segment_image.segmentImage('D:/Studia/Magisterskie/Magisterka/programy/Scale-recurrent Network for Deep Image Deblurring/SRN-Deblur/sample.jpeg',
    #                            extract_segmented_objects=True, save_extracted_objects=True, show_bboxes=True,  output_image_name='output.jpg')

    # ins = instanceSegmentation()
    # ins.load_model("pointrend_resnet50.pkl")
    # ins.segmentImage("D:/Studia/Magisterskie/Magisterka/programy/Scale-recurrent Network for Deep Image Deblurring/SRN-Deblur/01_deblur.png",
    #                  show_bboxes=True,extract_segmented_objects=True,  output_image_name="output_deblur.jpg")

    # change_bg = alter_bg(model_type="pb")
    # change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
    # change_bg.change_bg_img(f_image_path="D:/Studia/Magisterskie/Magisterka/programy/Scale-recurrent Network for Deep Image Deblurring/SRN-Deblur/02_deblur2.png",
    #                         b_image_path="D:/Studia/Magisterskie/Magisterka/programy/Scale-recurrent Network for Deep Image Deblurring/SRN-Deblur/02_blur.png",
    #                         output_image_name="02_new_img.jpg")

if __name__ == '__main__':
    main()

    # img = cv2.imread(
    #     "D:/Studia/Magisterskie/Magisterka/programy/Scale-recurrent Network for Deep Image Deblurring/SRN-Deblur/02_deblur2.png")
    # b_img = cv2.imread("D:/Studia/Magisterskie/Magisterka/programy/Scale-recurrent Network for Deep Image Deblurring/SRN-Deblur/02_blur2.png")
    # change_bg = alter_bg(model_type="pb")
    # change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
    # change_bg.change_bg_img(img, b_img, [0], output_image_name="02_new_img2.jpg")
