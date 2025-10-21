import os
import numpy as np
import time
import SimpleITK as sitk
import warnings
from glob import glob
import re
from util import *

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(itkimage.GetOrigin())
    numpySpacing = list(itkimage.GetSpacing())
    return numpyImage, numpyOrigin, numpySpacing

def load_dcm_image(Path,multides):
    if multides==False:
        s = [
            pydicom.read_file(os.path.join(Path, s), force=True)
            for s in os.listdir(Path)
        ]
        s.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        print('共', len(s), '张')
        ConstPixelDims = (int(s[0].Rows), int(s[0].Columns), len(s))
        Array = np.zeros(ConstPixelDims,
                            dtype=s[0].pixel_array.dtype)  # uint16类型
        for n in range(len(s)):
            Array[:, :, n] = s[n].pixel_array
        #     c.writelines([descri,'\n'])
        pz = np.abs(s[0].ImagePositionPatient[2] - s[1].ImagePositionPatient[2])
        px = float(s[0].PixelSpacing[0])
        py = float(s[0].PixelSpacing[1])
        spacing = (px, py, pz)
        print('像素大小为', spacing)
        origin = tuple(np.array(s[0].ImagePositionPatient))
        print('原点为', origin)
        intercept = s[0].RescaleIntercept  # 缩放截距 -1024
        slope = s[0].RescaleSlope  # 缩放斜率 1
        Array = Array * int(slope) + int(intercept) * np.ones(
            ConstPixelDims, dtype=s[0].pixel_array.dtype)
    else:
        slices = [pydicom.read_file(os.path.join(Path, s), force=True) for s in os.listdir(Path)]
        for si in range(len(slices)):
            if hasattr(slices[si], 'SeriesDescription'):
                ss_set=set(slices[si].SeriesDescription)
                break
        for ss in slices[1:]:
            if hasattr(ss, 'SeriesDescription'):
                ss_set.add(ss.SeriesDescription)
        ss_list=list(ss_set)
        ss_list.sort()
        for i in range(len(ss_list)):
            print(i,':',ss_list[i], '\n', end="")
        nice=0
        while nice ==0:
            des = int(input('请选择第几个描述：'))
            s=[x for x in slices if (hasattr(x, 'SeriesDescription')) and (x.SeriesDescription == ss_list[des])]
            s.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            print('共',len(s),'张')
            if len(s)==0:
                continue
            ConstPixelDims = (int(s[0].Rows), int(s[0].Columns), len(s))
            Array = np.zeros(ConstPixelDims, dtype=s[0].pixel_array.dtype)  # uint16类型
            for n in range(len(s)):
                Array[:, :, n] = s[n].pixel_array
            pylab.figure(1)
            pylab.imshow(Array[:,200,:],"gray")
            pylab.ion()
            pylab.pause(2)  #显示秒数
            pylab.close()
            nice = int(input('确定选择这个描述：'))
        pz = np.abs(s[0].ImagePositionPatient[2] - s[1].ImagePositionPatient[2])
        px = float(s[0].PixelSpacing[0])
        py = float(s[0].PixelSpacing[1])
        spacing=(px,py,pz)
        print('像素大小为',spacing)
        origin = tuple(np.array(s[0].ImagePositionPatient))
        print('原点为', origin)
        intercept = s[0].RescaleIntercept  # 缩放截距 -1024
        slope = s[0].RescaleSlope  # 缩放斜率 1
        Array = Array * int(slope) + int(intercept) * np.ones(ConstPixelDims, dtype=s[0].pixel_array.dtype)
    return Array,origin,spacing

def save_itk(image, origin, spacing, filename):
    """
	:param image: images to be saved
	:param origin: CT origin
	:param spacing: CT spacing
	:param filename: save name
	:return: None
	:param image:要保存的图像
	:param原点:CT原点
	:param spacing:CT间距
	:param filename:保存名称
	:return:无
	"""

    itkimage = sitk.GetImageFromArray(image)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename)

def savenpy(data_path, prep_folder, format='nii.gz', mode='prepro', multides=True):
    """
	:param data_path: input CT data path
	:param prep_folder:
	:return: None
	:param data_path:输入CT数据路径
	:param prep_folder:
	:return:无
	"""
    resolution = np.array([1, 1, 1])
    # name = data_path.split('/')[-1].split('.nii')[0]
    name = data_path.split('/')[-1].split('data.nii')[0]
    assert (os.path.exists(data_path) is True)
    if format == 'nii.gz':
        case_pixels, origin, spacing = load_itk_image(data_path)
        if spacing[0] != spacing[1] and spacing[1] == spacing[2]:
            px, py, pz = spacing[1], spacing[2], spacing[0]
            spacing = (px, py, pz)
            ox, oy, oz = origin[1], origin[2], origin[0]
            origin = (ox, oy, oz)
        a = case_pixels.shape[0]
        b = case_pixels.shape[1]
        c = case_pixels.shape[2]
        if b == c:
            case_pixels = case_pixels.transpose(1, 2, 0)
        elif a == c:
            case_pixels = case_pixels.transpose(0, 2, 1)
    if format == 'dcm':
        case_pixels, origin, spacing = load_dcm_image(data_path, multides)

    case_pixels=case_pixels+1024
    cmin = np.min(case_pixels)
    cmax = np.max(case_pixels)
    # hist = pylab.hist(case_pixels.flatten(), 300)
    hist =np.histogram(case_pixels.flatten(), 300)

    th = -800
    if cmin <= th:
        hhh1 = hist[1][np.where(hist[1] >= th)[0][0]:]
        hhh0 = hist[0][np.where(hist[1] >= th)[0][0]:]
        maxloc = np.where(hhh0 == np.max(hhh0))
        # print(maxLoc)
        firstPeak = hhh1[maxloc[0][0]]  # 灰度值
        measureDists = np.zeros([300], np.float32)
        for k in range(hhh0.shape[0]):
            measureDists[k] = pow(hhh1[k + 1] - firstPeak,
                                  2) * hhh0[k]  # 综合考虑 两峰距离与峰值
        maxloc2 = np.where(measureDists == np.max(measureDists))
        secondPeak = hhh1[maxloc2[0][0]]  # 灰度值
        aaa = firstPeak
        if secondPeak < firstPeak:
            aaa = secondPeak
        case_pixels[np.where(
            case_pixels <=
            th)] = aaa  
        cmax = np.max(case_pixels)



    if mode == 'prepro':
        T = th_2t(case_pixels)
        L = get_l(case_pixels, T)
        L1 = maximum_3d(L)
        L2 = maximum_3d(L ^ L1)
        L = L1 + L2

        Mask = L
        xx, yy, zz = np.where(Mask)
        box = np.array([[np.min(xx), np.max(xx)], [np.min(yy),
                                                np.max(yy)],
                        [np.min(zz), np.max(zz)]])
        margin = 5
        box = np.vstack([
            np.max([[0, 0, 0], box[:, 0] - margin], 0),
            np.min([np.array(Mask.shape), box[:, 1] + margin], axis=0).T
        ]).T

        # save the lung mask
        data_savepath = os.path.join(prep_folder, name + '_lung_mask.nii.gz')
        Mask_crop = Mask[box[0, 0]:box[0, 1], box[1, 0]:box[1, 1],
                        box[2, 0]:box[2, 1]]
        save_itk(Mask_crop.astype(dtype='uint8'), origin, spacing, data_savepath)

        sliceim_hu = case_pixels
        shapeorg = sliceim_hu.shape
        box_shape = np.array([[0, shapeorg[0]], [0, shapeorg[1]], [0,
                                                                shapeorg[2]]])
        sliceim2_hu = sliceim_hu[box[0, 0]:box[0, 1], box[1, 0]:box[1, 1],
                                box[2, 0]:box[2, 1]]
        box = np.concatenate([box, box_shape], axis=0)
        np.save(os.path.join(prep_folder, name + '_box.npy'), box)

    # save processed image
    data_savepath = os.path.join(prep_folder, name + 'data_cut.nii.gz')
    if mode == 'prediction':
        save_itk(case_pixels, origin, spacing, data_savepath)
    else:
        save_itk(sliceim2_hu, origin, spacing, data_savepath)
    return

def cutmask(data_path, prep_folder):
    name = data_path.split('CASE')[-1].split('mask')[0]
    assert (os.path.exists(data_path) is True)
    mask_p, origin, spacing = load_itk_image(data_path)
    # save box (image original shape and cropped window region)
    box = np.load(os.path.join(prep_folder[:-4]+'data','CASE'+name+'_box.npy'),allow_pickle=True)
    # mask_p = large_connected_domain(mask_p)
    mask_p = large_connected_domain26(mask_p)
    sliceim2_hu = mask_p[box[0, 0]:box[0, 1],
         box[1,0]:box[1,1],
         box[2,0]:box[2,1]]

    data_savepath = os.path.join(prep_folder, 'CASE'+name+'mask_cut.nii.gz')
    save_itk(sliceim2_hu.astype(dtype='uint8'), origin, spacing, data_savepath)

    return

def preprocess_CT(inputpath=None, savepath=None,format='nii.gz',mode='prepro',multides=True):
    """
	Preprocess the CT images in the input path to extract lung field
	Save the processed images in the save path
	:param inputpath: input data path
	:param savepath: output save path
	:return: save directory path
	:对输入路径中的CT图像进行预处理以提取肺野
	:在保存路径中保存处理后的图像
	:param inputpath:输入数据路径
	:param savepath:输出保存路径
	:return:保存目录路径
	"""
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    if format=='nii.gz':
        if mode=='prepro':
            filelist = glob(os.path.join(inputpath, '*.nii*'))  # default nifty format
            filelist.sort()
            print (inputpath, filelist)
            for curfilepath in filelist:######################################################################################
                start_time = time.time()
                print ('starting preprocessing lung CT')
                savenpy(data_path=curfilepath, prep_folder=savepath,format=format,mode=mode)
                end_time = time.time()
                print ('end preprocessing lung CT, time %d seconds'%(end_time-start_time))
        if mode=='prediction':
            start_time = time.time()
            print ('starting preprocessing lung CT')
            savenpy(data_path=inputpath, prep_folder=savepath,format=format,mode=mode)
            end_time = time.time()
            print ('end preprocessing lung CT, time %d seconds'%(end_time-start_time))

    if format=='dcm':
        start_time = time.time()
        print ('starting preprocessing lung CT')
        savenpy(data_path=inputpath, prep_folder=savepath,format=format,mode=mode,multides=multides)
        end_time = time.time()
        print ('end preprocessing lung CT, time %d seconds'%(end_time-start_time))



    return savepath

def preprocess_mask(inputpath=None, savepath=None):
    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    filelist = glob(os.path.join(inputpath, '*.nii*'))  # default nifty format
    print (inputpath, filelist)
    for curfilepath in filelist:
        start_time = time.time()
        print ('starting preprocessing lung mask')
        cutmask(data_path=curfilepath, prep_folder=savepath)
        end_time = time.time()
        print ('end preprocessing lung mask, time %d seconds'%(end_time-start_time))

    return savepath


if __name__ == '__main__':

    inputpath = '/mnt/yby/AIIB23_Train_T1/img'
    savepath = '/mnt/yby/AFTER_PREPROCESS/data'
    inputpath2 = '/mnt/yby/AIIB23_Train_T1/gt'
    savepath2 = '/mnt/yby/AFTER_PREPROCESS/mask'

    preprocess_CT(inputpath, savepath)
    preprocess_mask(inputpath2, savepath2)

