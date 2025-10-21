import argparse
import os
import numpy as np
import skimage.measure as measure
import time
import pyvista as pv
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize_3d
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path

from util import *
from atm22_skel_parse import *
from ours_skel_parse import Topology_Tree


def ours_skel_parse(pred, spacing, merge_t, save_dir, case):
    start_time = time.time()
    px, py, pz = spacing[0], spacing[1], spacing[2]
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
    airway_topo.regrade()
    if airway_topo.rb23==1 or airway_topo.rb12==1 :
        airway_topo.remerge()
        airway_topo.regrade()

    end_time = time.time()
    centerline_time = end_time - start_time
    print('Centerline segment time %d seconds' % centerline_time)

    # Save airway_topo.Bi_resize a .npy file
    airway_topo.resize(px, py, pz,os.path.join(save_dir, case.split('.nii.gz')[0] + '_parse.npy'))  #Perform resizing operation using input pixel size to obtain airway_topo.Bi_resize
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dir_r = os.path.join(save_dir, case.split('.nii.gz')[0] + '.stl')
    airway_topo.recons(dir_r)
    airway_topo.show_line1(save_dir, case)  #Visualize centerline segments
    start_time = time.time()

    end_time=airway_topo.sub_model(start_time, save_dir, case)  #Tree parse
    tree_parse_time = end_time - start_time
    print('Airway tree parse time %d seconds' % tree_parse_time)
    print('Number of branches %d ' % (len(airway_topo.Bi)))

    # Save time information to a text file
    time_filename = os.path.join(save_dir, case.split('.nii.gz')[0] + '_time.txt')
    with open(time_filename, 'w') as f:
        f.write('Centerline segment time %d seconds\n' % centerline_time)
        f.write('Airway tree parse time %d seconds\n' % tree_parse_time)
        f.write('Number of branches %d\n' % (len(airway_topo.Bi)))

    return airway_topo

def atm22_skel_parse(label, spacing, olddir, case):
    colors = []
    color_list = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
        'E', 'F'
    ]
    for i in range(400):
        color = ''
        for i in range(6):
            color_number = color_list[random.randint(0, 15)]
            color += color_number
        color = '#' + color
        colors.append(color)

    start_time = time.time()

    #recons
    label = large_connected_domain(label)
    iso = 0.95
    verts, faces, _, _ = marching_cubes(label, iso)
    verts[:, 0] = verts[:, 0] * spacing[0]
    verts[:, 1] = verts[:, 1] * spacing[1]
    verts[:, 2] = verts[:, 2] * spacing[2]
    faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
    mesh_airway = pv.PolyData(verts, faces)
    mesh_airway_smooth = mesh_airway.smooth(relaxation_factor=0.2)
    pl = pv.Plotter()
    pl.add_mesh(mesh_airway_smooth, color='#E96C6F', style='surface')
    if not os.path.exists(olddir):
        os.mkdir(olddir)
    dir_r = os.path.join(olddir, case.split('.nii.gz')[0] + '.stl')
    mesh_airway_smooth.save(dir_r)

    #Centerline segment/ tuning
    skeleton = skeletonize_3d(label)
    skeleton_parse, cd, num = skeleton_parsing(skeleton)
    end_time = time.time()
    centerline_time = end_time - start_time
    print('Centerline segment time %d seconds' % centerline_time)
    mesh_airway_smooth = pv.read(dir_r)
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh_airway_smooth,
                color='white',
                style='surface',
                opacity=0.4)
    for i in range(1, num + 1):
        bi = np.array(np.where(cd == i))
        bi = bi[:, bi[2].argsort()]
        bi = bi.T
        bi = np.array(bi)
        if bi.shape[0] % 2 != 0:
            bi = bi[:-1, :]
        bi = bi.astype(np.float32)
        bi[:, 0] *= spacing[0]
        bi[:, 1] *= spacing[1]
        bi[:, 2] *= spacing[2]

        pl.add_lines(bi, color=colors[i - 1], width=5)
    pl.background_color = "white"
    pl.view_yz()
    screenshot_filename = os.path.join(olddir,
                                       case.split('.nii.gz')[0] + '.png')
    pl.screenshot(screenshot_filename)
    pl.close()

    #Tree parse/ tuning
    start_time = time.time()
    tree_parsing = tree_parsing_func(skeleton_parse, label, cd)
    trachea = loc_trachea(tree_parsing, num)
    ad_matric = adjacent_map(tree_parsing, num)
    parent_map, children_map, generation = parent_children_map(
        ad_matric, trachea, num)
    while whether_refinement(parent_map, children_map, tree_parsing, num,
                             trachea) is True:
        tree_parsing, num = tree_refinement(parent_map, children_map,
                                            tree_parsing, num, trachea)
        trachea = loc_trachea(tree_parsing, num)
        ad_matric = adjacent_map(tree_parsing, num)
        parent_map, children_map, generation = parent_children_map(
            ad_matric, trachea, num)

    end_time = time.time()
    tree_parse_time = end_time - start_time
    print('Airway tree parse time %d seconds' % tree_parse_time)
    print('Number of branches %d ' % (num))
    pl = pv.Plotter(off_screen=True)
    pl.open_gif(os.path.join(olddir, case.split('.nii.gz')[0] + '.gif'))
    for k in range(1, num + 1):
        iso = 0.95
        verts, faces, _, _ = marching_cubes(tree_parsing == k, iso)
        verts[:, 0] = verts[:, 0] * spacing[0]
        verts[:, 1] = verts[:, 1] * spacing[1]
        verts[:, 2] = verts[:, 2] * spacing[2]
        faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
        mesh_seg = pv.PolyData(verts, faces)
        mesh_seg = mesh_seg.smooth(relaxation_factor=0.15)
        pl.add_mesh(mesh_seg, color=colors[k - 1], style='surface')  #
    n_frames = 60
    for i in range(n_frames):
        pl.camera_position = 'yz'
        pl.camera.azimuth = 360 * (i / n_frames)
        pl.render()
        pl.write_frame()
    pl.close()


    pl = pv.Plotter(off_screen=True)
    for k in range(1, num + 1):
        iso = 0.95
        verts, faces, _, _ = marching_cubes(tree_parsing == k, iso)
        verts[:, 0] = verts[:, 0] * spacing[0]
        verts[:, 1] = verts[:, 1] * spacing[1]
        verts[:, 2] = verts[:, 2] * spacing[2]
        faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
        mesh_seg = pv.PolyData(verts, faces)
        mesh_seg = mesh_seg.smooth(relaxation_factor=0.15)
        pl.add_mesh(mesh_seg, color=colors[k - 1], style='surface')  #

    pl.camera_position = 'yz'
    pl.show(screenshot=os.path.join(olddir, case.split('.nii.gz')[0] + '_model.png'))


    # Save time information to a text file
    time_filename = os.path.join(olddir, case.split('.nii.gz')[0] + '_time.txt')
    with open(time_filename, 'w') as f:
        f.write('Centerline segment time %d seconds\n' % centerline_time)
        f.write('Airway tree parse time %d seconds\n' % tree_parse_time)
        f.write('Number of branches %d\n' % num)


    return tree_parsing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Process airway segmentation and tree parsing.")

    parser.add_argument(
        '--pred_mask_path',
        type=str,
        default='./demo_mask/',
        help="Path to the directory containing predicted mask files.")
    parser.add_argument('--save_path',
                        type=str,
                        default=None,
                        help="Directory where the Ours output will be saved.")
    parser.add_argument('--save_ATM22_path',
                        type=str,
                        default=None,
                        help="Directory where the ATM22 output will be saved.")
    parser.add_argument(
        '--merge_t',
        type=int,
        default=5,
        help="Threshold for merging branches during airway skeleton parsing.")

    args = parser.parse_args()

    pred_mask_path = args.pred_mask_path
    save_path = args.save_path
    save_ATM22_path = args.save_ATM22_path
    merge_t = args.merge_t

    # pred_mask_path = './demo_mask/'
    # save_path ='./demo_output_Ours/'
    # save_ATM22_path = './demo_output_ATM22/'
    # merge_t = 5

    flist = os.listdir(pred_mask_path)
    flist.sort()

    for case in flist:
        pred, _, spacing = load_itk_image(os.path.join(pred_mask_path, case))

        if save_path is not None:
            if not Path(save_path).exists():
                Path(save_path).mkdir(parents=True, exist_ok=True)
            airway_topo = ours_skel_parse(pred, spacing, merge_t, save_path,case)

        if save_ATM22_path is not None:
            if not Path(save_ATM22_path).exists():
                Path(save_ATM22_path).mkdir(parents=True, exist_ok=True)
            tree_parsing = atm22_skel_parse(pred, spacing, save_ATM22_path,case)
