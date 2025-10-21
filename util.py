import numpy as np
import skimage.measure as measure
import random
import cc3d
from scipy.ndimage.morphology import binary_fill_holes
import SimpleITK as sitk


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
