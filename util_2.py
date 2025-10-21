import os
import cv2
import numpy as np
import pylab
import skimage.measure as measure
import pydicom
import random
import csv
import codecs
import cc3d
import pynvml
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img

def size_calc(Path):
    slices = [pydicom.read_file(os.path.join(Path, s), force=True) for s in os.listdir(Path)]
    # 按z轴排序切片
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    # 计算切片厚度
    pz = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    px = float(slices[0].PixelSpacing[0])
    py = float(slices[0].PixelSpacing[1])
    return px, py, pz

def th_2t(a,format='dcm'):
    kmax=300
    if format=='jpg':
        kmax=50
    # hist = pylab.hist(a.flatten(), kmax)
    hist=np.histogram(a.flatten(), kmax)
    # pylab.ion()
    # pylab.pause(2)  #显示秒数
    # pylab.close()
    hist_y = hist[0]
    hist_x = hist[1]
    # 寻找灰度直方图的最大峰值对应的灰度值
    maxloc = np.where(hist_y == np.max(hist_y))
    # print(maxLoc)
    firstPeak = hist_x[maxloc[0][0]]  # 灰度值
    # 寻找灰度直方图的第二个峰值对应的灰度值
    measureDists = np.zeros([300], np.float32)
    for k in range(kmax):
        measureDists[k] = pow(hist_x[k + 1] - firstPeak, 2) * hist_y[k]  # 综合考虑 两峰距离与峰值
    maxloc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = hist_x[maxloc2[0][0]]  # 灰度值
    # 找到两个峰值之间的最小值对应的灰度值，作为阈值
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
    L = np.zeros(Asizes, dtype=np.uint8)  # 记录肺区分割标记
    for n in range(int(0.05 * Asizes[2]) - 1, int(0.95 * Asizes[2])):
        A = np.zeros((Asizes[0], Asizes[1]), dtype=int)
        for i in range(Asizes[0]):
            for j in range(Asizes[1]):
                if Array[i, j, n] >= T:
                    A[i, j] = 1
        imLabel, _ = measure.label(A, background=0, return_num=True)  # 对各连通域进行标记,计数连通域
        noback = np.bincount(imLabel.reshape(-1))  # 剔除背景干扰
        noback[0] = 0
        max_label = np.argmax(noback)  # 获得个数最多的值的下标（就是值本身）
        img1 = (imLabel == max_label)  # 获取最大连通域图像
        img2 = binary_fill_holes(img1)
        img3 = img1 ^ img2
        imLabel, _ = measure.label(img3, background=0, return_num=True)
        noback = np.bincount(imLabel.reshape(-1))
        noback[0] = 0
        max_label1 = np.argmax(noback)  # 获得个数最多的值的下标（就是值本身）
        max_num1 = np.max(noback)
        img11 = (imLabel == max_label1)  # 这是第一大连通域图像（不确定是否大于2000，需验证后才能作为肺实质）
        noback[max_label1] = 0
        max_label2 = np.argmax(noback)  # 获得个数最多的值的下标（就是值本身）
        max_num2 = np.max(noback)
        img22 = (imLabel == max_label2)  # 这是第二大连通域图像（不确定是否大于2000，需验证后才能作为肺实质）
        if max_num1 > 2000:
            L[:, :, n] = img11
        if max_num2 > 2000:
            L[:, :, n] = L[:, :, n] | img22
        # pylab.subplot(221)
        # pylab.imshow(Array[:, :, n],"gray")
        # pylab.subplot(222)
        # pylab.imshow(A,"gray")
        # pylab.subplot(223)
        # pylab.imshow(img3,"gray")
        # pylab.subplot(224)
        # pylab.imshow(L[:, :, n],"gray")
        # pylab.show()
    return L

def maximum_3d(region01):
    # 标记输入的3D图像
    # label, num = measure.label(region01, connectivity=1, return_num=True)
    label=cc3d.connected_components(region01, connectivity=26)
    num=np.max(label)
    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取一块区域面积并排序
    num_list = [i for i in range(1, num+1)]
    area_list = [region[i-1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    max_region01=(label==num_list_sorted[0])
    if max_region01[:,:,region01.shape[2]//2].any()==0 and  max_region01[:,:,region01.shape[2]//3].any()==0 and  max_region01[:,:,region01.shape[2]//3*2].any()==0:
        max_region01=(label==num_list_sorted[1])
    max_region01=max_region01.astype(np.int8)
    max_region01=binary_fill_holes(max_region01)
    return max_region01

def random_color():
    color_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                  'A', 'B', 'C', 'D', 'E', 'F']
    color = ''
    # 一个以“#”开头的6位十六进制数值表示一种颜色，所以要循环6次
    for i in range(6):
        # random.randint表示产生0~15的一个整数型随机数
        color_number = color_list[random.randint(0, 15)]
        color += color_number
    color = '#' + color
    return color

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name,'w+','utf-8')#追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")

def find_bb_3D(label):
    if len(label.shape) != 3:
        print("The dimension of input is not 3!")
        os._exit()
    sum_x = np.sum(label, axis=(1, 2))
    sum_y = np.sum(label, axis=(0, 2))
    sum_z = np.sum(label, axis=(0, 1))
    xf = np.where(sum_x)
    xf = xf[0]
    yf = np.where(sum_y)
    yf = yf[0]
    zf = np.where(sum_z)
    zf = zf[0]
    x_length = xf.max() - xf.min() + 1
    y_length = yf.max() - yf.min() + 1
    z_length = zf.max() - zf.min() + 1
    x1 = xf.min()
    y1 = yf.min()
    z1 = zf.min()

    cs = [x_length + 8, y_length + 8, z_length + 8]
    for j in range(3):
        if cs[j] > label.shape[j]:
            cs[j] = label.shape[j]
    # print(cs[0], x_length)
    # x_length, y_length, z_length, x1, y1, z1 = find_bb_3D(label2)
    cs = np.array(cs, dtype=np.uint16)
    size = label.shape
    xl = x1 - (cs[0] - x_length) // 2
    yl = y1 - (cs[1] - y_length) // 2
    zl = z1 - (cs[2] - z_length) // 2
    xr = xl + cs[0]
    yr = yl + cs[1]
    zr = zl + cs[2]
    if xl < 0:
        xl = 0
        xr = cs[0]
    if xr > size[0]:
        xr = size[0]
        xl = xr - cs[0]
    if yl < 0:
        yl = 0
        yr = cs[1]
    if yr > size[1]:
        yr = size[1]
        yl = yr - cs[1]
    if zl < 0:
        zl = 0
        zr = cs[2]
    if zr > size[2]:
        zr = size[2]
        zl = zr - cs[2]
    return xl, xr, yl, yr, zl, zr

def large_connected_domain(label):
    cd, num = measure.label(label, return_num=True, connectivity=1)
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((cd == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    label = (cd == (volume_sort[-1] + 1)).astype(np.uint8)
    label = ndimage.binary_fill_holes(label)
    label = label.astype(np.uint8)
    return label

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

def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """

    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    # return total, used, free
    return free
