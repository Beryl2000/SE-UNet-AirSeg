import numpy as np
import skimage.measure as measure
import random
import cc3d
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
import SimpleITK as sitk
import pynvml


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = list(itkimage.GetOrigin())
    numpySpacing = list(itkimage.GetSpacing())
    a = numpyImage.shape[0]
    b = numpyImage.shape[1]
    c = numpyImage.shape[2]
    if b == c:
        return numpyImage.transpose(1, 2, 0), numpyOrigin, numpySpacing
    elif a == b:
        return numpyImage, numpyOrigin, numpySpacing

    
def random_color():
    color_list = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
        'E', 'F'
    ]
    color = ''
    for i in range(6):
        color_number = color_list[random.randint(0, 15)]
        color += color_number
    color = '#' + color

    return color

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


def maximum_3d(region01):
    label = cc3d.connected_components(region01, connectivity=26)
    num = np.max(label)
    region = measure.regionprops(label)
    num_list = [i for i in range(1, num + 1)]
    area_list = [region[i - 1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x - 1])[::-1]
    max_region01 = (label == num_list_sorted[0])
    if max_region01[:, :, region01.shape[2] // 2].any(
    ) == 0 and max_region01[:, :, region01.shape[2] //
                            3].any() == 0 and max_region01[:, :,
                                                           region01.shape[2] //
                                                           3 * 2].any() == 0:
        max_region01 = (label == num_list_sorted[1])
    max_region01 = max_region01.astype(np.int8)
    max_region01 = binary_fill_holes(max_region01)

    return max_region01


def get_gpu_mem_info(gpu_id=0):

    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} not exist!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    # return total, used, free
    return free



def th_2t(a,format='dcm'):
    kmax=300
    if format=='jpg':
        kmax=50
    hist=np.histogram(a.flatten(), kmax)
    hist_y = hist[0]
    hist_x = hist[1]
    maxloc = np.where(hist_y == np.max(hist_y))
    # print(maxLoc)
    firstPeak = hist_x[maxloc[0][0]]  
    measureDists = np.zeros([300], np.float32)
    for k in range(kmax):
        measureDists[k] = pow(hist_x[k + 1] - firstPeak, 2) * hist_y[k]  
    maxloc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = hist_x[maxloc2[0][0]]  
    if maxloc2[0][0]>maxloc[0][0]:
        hist_y[maxloc2[0][0]::] = np.max(hist_y)
        hist_y[0:maxloc[0][0]] = np.max(hist_y)
    else:
        hist_y[maxloc[0][0]::] = np.max(hist_y)
        hist_y[0:maxloc2[0][0]] = np.max(hist_y)
    T = hist_x[np.where(hist_y == np.min(hist_y))[0][0]]
    return T


def get_l(Array,T):

    Asizes=Array.shape
    L = np.zeros(Asizes, dtype=np.uint8)
    for n in range(int(0.05 * Asizes[2]) - 1, int(0.95 * Asizes[2])):
        A = np.zeros((Asizes[0], Asizes[1]), dtype=int)
        for i in range(Asizes[0]):
            for j in range(Asizes[1]):
                if Array[i, j, n] >= T:
                    A[i, j] = 1
        imLabel, _ = measure.label(A, background=0, return_num=True) 
        noback = np.bincount(imLabel.reshape(-1))  
        noback[0] = 0
        max_label = np.argmax(noback) 
        img1 = (imLabel == max_label) 
        img2 = binary_fill_holes(img1)
        img3 = img1 ^ img2
        imLabel, _ = measure.label(img3, background=0, return_num=True)
        noback = np.bincount(imLabel.reshape(-1))
        noback[0] = 0
        max_label1 = np.argmax(noback) 
        max_num1 = np.max(noback)
        img11 = (imLabel == max_label1)  
        noback[max_label1] = 0
        max_label2 = np.argmax(noback) 
        max_num2 = np.max(noback)
        img22 = (imLabel == max_label2)  
        if max_num1 > 2000:
            L[:, :, n] = img11
        if max_num2 > 2000:
            L[:, :, n] = L[:, :, n] | img22

    return L



def large_connected_domain26(label):
    cd=cc3d.connected_components(label, connectivity=26)
    num=np.max(cd)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    label = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    label = ndimage.binary_fill_holes(label)
    label = label.astype(np.uint8)
    return label