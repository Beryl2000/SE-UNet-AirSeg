import numpy as np
import os
import nibabel
import copy
import cc3d
import pyvista as pv
from scipy import ndimage
from stl import mesh
import SimpleITK as sitk
import skimage.measure as measure
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize_3d,binary_closing,binary_dilation
from scipy.interpolate import interp1d
from data import load_json_file
from util import *
from ours_skel_parse import Topology_Tree, tree_parsing_func



def airway_parse(pred,merge_t):
    minz = np.min(np.where(pred == 1)[2])
    maxz = np.max(np.where(pred == 1)[2])
    cha = maxz - minz
    oneslice = pred[:, :, int(0.8 * cha + minz)]
    imLabel, _ = measure.label(oneslice, background=0, return_num=True)
    noback = np.bincount(imLabel.reshape(-1))
    noback[0] = 0
    max_num8 = np.max(noback)
    oneslice = pred[:, :, int(0.2 * cha + minz)]
    imLabel, _ = measure.label(oneslice, background=0, return_num=True)
    noback = np.bincount(imLabel.reshape(-1))
    noback[0] = 0
    max_num2 = np.max(noback)
    if max_num2 > max_num8:
        order = 0
    else:
        order = 1

    remerge_l=['000']
    airway_topo = Topology_Tree(pred, order, merge_t,remerge_l)
    airway_topo.sub(
    )  #Skeleton extraction and segmentation operations yield airway_topo.Bi
    airway_topo.merge(
    )  #Remove small segments in leaves and branches to update airway_topo.Bi

    airway_topo.grade()

    skeleton_parse = cd = np.zeros(airway_topo.label.shape, dtype=np.int32)
    iii = 1
    for i in airway_topo.Bi:
        bi = i['member'].copy()
        bi.insert(0, i['start'])
        if 'end' in i:
            bi.append(i['end'])
        bi = np.array(bi)
        for u in range(bi.shape[0]):
            if cd[bi[u, 0], bi[u, 1], bi[u, 2]] == 0:
                cd[bi[u, 0], bi[u, 1], bi[u, 2]] = iii
        iii += 1

    skeleton_parse = cd.copy()
    skeleton_parse[np.where(skeleton_parse != 0)] = 1
    tree_parsing = tree_parsing_func(skeleton_parse, airway_topo.label, cd)

    return tree_parsing

def tree_parse(label_root, file_path, save_root, save_root2):
    file_list = load_json_file(file_path, folder='0', mode=['train'])
    file_list.sort()

    for ids in range(len(file_list)):
        f = file_list[ids] + 'mask_cut.nii.gz'
        # if os.path.exists(os.path.join(save_root, f)):
        #     continue
        label = sitk.ReadImage(os.path.join(label_root, f))
        numpyOrigin = list(label.GetOrigin())
        numpySpacing = list(label.GetSpacing())
        label = sitk.GetArrayFromImage(label) 
        # label = label.transpose(2, 1, 0) 
        label = (label > 0).astype(np.uint8)
        label = large_connected_domain26(label)
        
        skeleton = skeletonize_3d(label)
        skeleton_nii = sitk.GetImageFromArray(skeleton)
        skeleton_nii.SetSpacing(numpySpacing)
        skeleton_nii.SetOrigin(numpyOrigin)
        sitk.WriteImage(skeleton_nii, os.path.join(save_root2, f))


        tree_parsing = airway_parse(label,merge_t=5)

        print(ids, '/', len(file_list), f, "finished!")
        parse_nii = sitk.GetImageFromArray(tree_parsing)
        parse_nii.SetSpacing(numpySpacing)
        parse_nii.SetOrigin(numpyOrigin)
        sitk.WriteImage(parse_nii,os.path.join(save_root, f))

def tree_parse_val(label_root,file_path,save_root,save_root2):
    file_list = load_json_file(file_path, folder='0', mode=['val'])
    file_list.sort()

    for ids in range(len(file_list)):
        f = file_list[ids] + 'mask_cut.nii.gz'

        label = sitk.ReadImage(os.path.join(label_root, f))
        numpyOrigin = list(label.GetOrigin())
        numpySpacing = list(label.GetSpacing())
        label = sitk.GetArrayFromImage(label)
        # label = label.transpose(2, 1, 0)
        label = (label > 0).astype(np.uint8)
        label = large_connected_domain26(label)
        # LABEL_TRANS = binary_closing(binary_fill_holes(binary_dilation(label)))
        
        # skeleton = skeletonize_3d(LABEL_TRANS)
        skeleton = skeletonize_3d(label)
        # skeleton = skeleton.transpose(2, 1, 0)
        skeleton_nii = sitk.GetImageFromArray(skeleton)
        skeleton_nii.SetSpacing(numpySpacing)
        skeleton_nii.SetOrigin(numpyOrigin)
        sitk.WriteImage(skeleton_nii, os.path.join(save_root2, f))


        tree_parsing = airway_parse(label, merge_t=5)

        print(ids, '/', len(file_list), f, "finished!")
        parse_nii = sitk.GetImageFromArray(tree_parsing)
        parse_nii.SetSpacing(numpySpacing)
        parse_nii.SetOrigin(numpyOrigin)
        sitk.WriteImage(parse_nii,os.path.join(save_root, f))

def tree_parse_test(label_root,file_path,save_root,save_root2):
    file_list = load_json_file(file_path, folder='-1', mode=['test'])
    file_list.sort()

    for ids in range(len(file_list)):
        f = file_list[ids] + 'mask_cut.nii.gz'
        # if os.path.exists(os.path.join(save_root, f)):
        #     continue
        label = sitk.ReadImage(os.path.join(label_root, f))
        numpyOrigin = list(label.GetOrigin())
        numpySpacing = list(label.GetSpacing())
        label = sitk.GetArrayFromImage(label)
        # label = label.transpose(2, 1, 0)
        label = (label > 0).astype(np.uint8)
        label = large_connected_domain26(label)
        # LABEL_TRANS = binary_closing(binary_fill_holes(binary_dilation(label)))
        
        # skeleton = skeletonize_3d(LABEL_TRANS)
        skeleton = skeletonize_3d(label)
        # skeleton = skeleton.transpose(2, 1, 0)
        skeleton_nii = sitk.GetImageFromArray(skeleton)
        skeleton_nii.SetSpacing(numpySpacing)
        skeleton_nii.SetOrigin(numpyOrigin)
        sitk.WriteImage(skeleton_nii, os.path.join(save_root2, f))


        tree_parsing = airway_parse(label, merge_t=5)

        print(ids, '/', len(file_list), f, "finished!")
        parse_nii = sitk.GetImageFromArray(tree_parsing)
        parse_nii.SetSpacing(numpySpacing)
        parse_nii.SetOrigin(numpyOrigin)
        sitk.WriteImage(parse_nii,os.path.join(save_root, f))




if __name__ == '__main__':

    label_root = 'AFTER_DATA/mask'
    file_path = './data/base_dict.json'
    save_root = './data/tree_parse'
    save_root2 = './data/skeleton'

    tree_parse(label_root, file_path, save_root, save_root2)

    label_root = 'AFTER_DATA/mask'
    file_path = './data/base_dict.json'
    save_root = './data/tree_parse_val'
    save_root2 = './data/skeleton_val'

    tree_parse_val(label_root, file_path, save_root, save_root2)

    label_root = 'AFTER_DATA/mask'
    file_path = './data/test.json'
    save_root = './data/tree_parse_test'
    save_root2 = './data/skeleton_test'

    tree_parse_test(label_root, file_path, save_root, save_root2)


