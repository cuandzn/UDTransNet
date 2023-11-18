import cv2
import os
import pydicom
import numpy as np
import copy
from pydicom.uid import ExplicitVRLittleEndian
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    return parser.parse_args()

# dcmtobmp.py  -->  predict.py --> CTseg.py
# CTseg.py: 原CT高斯滤波后分别设置阈值分出骨头和肋软骨。在肋软骨图片上进行三图组合：先肋软骨，在骨头（包括肋骨和钙化），最后钙化。
# 组合好的图片重新导成CT，三维重建
if __name__ == '__main__':
# ds = r"/root/data/rib dataset/18-30/0select"    # data s
# ms = rs = r"/root/data/UDtransnet_dataset/19-30/0select"   # result s
    args = get_args()
    ds = ''.join(args.input)
    ms = rs = ''.join(args.output)
    orderlist = os.listdir(ds)
    orderlist.sort()
    for i in range(0, len(orderlist)):    # 遍历CT文件夹
        ctlist = os.listdir(os.path.join(ds,orderlist[i]))    # 一个病人取一个CT组
        ctlist.sort()
        dpath = ctlist[0]
        dcmpath = os.path.join(ds + str('/') + orderlist[i],dpath)
        maskpath = os.path.join(ms + str('/') + orderlist[i],"con_mask")
        masklist = os.listdir(maskpath)
        masklist.sort(key=lambda x: int(x.split('.bmp')[0]))
        dcmlist = os.listdir(dcmpath)
        dcmlist.sort(key=lambda x: int(x.split('_')[-2]))
        outdcmpath = os.path.join(os.path.join(rs,orderlist[i]),"con_outdcm")
        if (os.path.exists(outdcmpath) == 0):  # costal保存路径不存在则创建路径
            os.makedirs(outdcmpath)
        for j in range(len(dcmlist)):   # 遍历CT
            if os.path.exists(os.path.join(dcmpath, dcmlist[j])):
                dcm = copy.copy(pydicom.dcmread(os.path.join(dcmpath, dcmlist[j])))
            else:
                print("dcm not exist")
            mask = cv2.imread(os.path.join(maskpath,str(len(dcmlist) - j - 1) + ".bmp"),0)
            # print(os.path.join(maskpath,str(len(dcmlist) - j - 1) + ".bmp"))
            # for x in range(512):
            #     for y in range(512):
            #         (r,g,b)=mask[x,y]
            #         if r!=g:
            #             print(r,g,b)
            pixel_arr = dcm.pixel_array  # 直接从CT中读出dcm的像素值数组
            pixel_arr = np.uint16(pixel_arr)
            blurimg = cv2.GaussianBlur(pixel_arr, (5,5), 0)
            # bone二值化：五像素高斯滤波 二值化阈值225+1024
            # Hu = pixel * slope(1) + intercept(-1024)  像素值-->CT值
            bonethre = 225      # 骨骼的CT范围(225,max)

            retbone, boneimg = cv2.threshold(blurimg, bonethre + 1024, 255, cv2.THRESH_BINARY)
            boneimg = np.uint8(boneimg)  # 图像归一化     outimg=outimg/255*255
            # costal cartilage二值化 五像素高斯滤波 二值化阈值70+1024,226+1024 (70,225]
            threshold_low = 70
            threshold_high = 225
            retcos0, low = cv2.threshold(blurimg, threshold_low + 1024, 100, cv2.THRESH_BINARY)
            retcos1, high = cv2.threshold(blurimg, threshold_high + 1024, 100, cv2.THRESH_BINARY_INV)
            cosimg = cv2.bitwise_and(low, high)
            cosimg = np.uint8(cosimg)  # 图像归一化  outimg=outimg/255*255
            cosimg = cv2.morphologyEx(cosimg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            # 组合
            outimg = cosimg
            # udtransnet
            # for x in range(outimg.shape[0]):
            #     for y in range(outimg.shape[1]):
            #         if boneimg[x][y] == 255:
            #             outimg[x][y] = 255
            #         # print(mask[x,y])
            #         (r,g,b) = mask[x,y]            
            #         if (r,g,b) == (0,0,255):
            #             #print("yes----------")
            #             outimg[x][y] = 180
            
            # unet
            for x in range(outimg.shape[0]):
                for y in range(outimg.shape[1]):
                    if boneimg[x][y] == 255:
                        outimg[x][y] = 255
                    if mask[x][y] == 63:
                        outimg[x][y] = 180
            img16 = np.array(outimg,dtype=np.int16)
            dcm.PixelData = img16.tobytes()    # len(pd)  # 524288
            dcm.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dcm.save_as(os.path.join(outdcmpath,str(len(dcmlist)-j-1)+".dcm"))
            print(os.path.join(outdcmpath,str(len(dcmlist)-j-1)+".dcm"))