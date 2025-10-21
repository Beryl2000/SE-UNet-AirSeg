import numpy as np
import copy
import pyvista as pv
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize_3d,binary_closing,binary_dilation
from scipy.ndimage.morphology import binary_fill_holes
from scipy.interpolate import interp1d
from scipy import ndimage
import os
import time

from util import *


class NDSparseMatrix:
    def __init__(self):
        self.elements = {}

    def addValue(self, tuple, value) -> object:
        self.elements[tuple] = value

    def readValue(self, tuple):
        try:
            value = self.elements[tuple]
        except KeyError:
            # could also be 0.0 if using floats...
            value = 0
        return value

def subsection(skeil, debug=0):
    """
    Segments a skeletonized 3D structure into branches and stores information about each branch.

    Args:
        skeil (ndarray): 3D array representing the skeleton.
        debug (int, optional): If non-zero, enables debug mode for additional processing steps. Defaults to 0.

    Returns:
        list: A list of dictionaries, each containing information about a branch (index, start, end, member points, father index).
    
    Updates:
        - branch_dict: List of segmented branches with detailed information.
    """
    skeilvol = NDSparseMatrix()
    for i in range(0, len(skeil)):
        skeilvol.addValue((skeil[i, 0], skeil[i, 1], skeil[i, 2]), 1)
    neigb = [[-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0], [0, 1, 0],
             [1, -1, 0], [1, 0, 0], [1, 1, 0], [-1, -1, -1], [-1, 0, -1],
             [-1, 1, -1], [0, -1, -1], [0, 0, -1], [0, 1, -1], [1, -1, -1],
             [1, 0, -1], [1, 1, -1], [-1, -1, 1], [-1, 0, 1], [-1, 1, 1],
             [0, -1, 1], [0, 0, 1], [0, 1, 1], [1, -1, 1], [1, 0, 1],
             [1, 1, 1]]
    index = np.where(skeil[:, 2] == np.min(skeil[:, 2]))
    s1 = skeil[index[0][0]]
    branchn = 0
    branch_dict = [] # List to store branch information
    visited = NDSparseMatrix() # To track visited points
    visited.addValue((s1[0], s1[1], s1[2]), 1)
    startnode = [[s1[0], s1[1], s1[2], 0]]

    while startnode:
        branch_dictn = {}
        branchn = branchn + 1
        branch_dictn['index'] = branchn
        branch_dictn['start'] = [
            startnode[0][0], startnode[0][1], startnode[0][2]
        ]
        linkstack = []
        member = []
        for i in range(0, 26):
            xi = startnode[0][0] + neigb[i][0]
            yi = startnode[0][1] + neigb[i][1]
            zi = startnode[0][2] + neigb[i][2]
            if skeilvol.readValue((xi, yi, zi)) and not (visited.readValue(
                (xi, yi, zi))):
                linkstack.append([xi, yi, zi])
        if linkstack.__len__() > 1:
            flag = 0
            for l in range(1, linkstack.__len__()):
                branch_dictn = {}
                branchn = branchn + l - 1
                branch_dictn['index'] = branchn
                branch_dictn['start'] = [
                    startnode[0][0], startnode[0][1], startnode[0][2]
                ]
                linkstackk = [linkstack[l - flag]]
                while linkstackk:
                    count = 0
                    neigb_skeil = []
                    for j in range(0, 26):
                        xj = linkstackk[0][0] + neigb[j][0]
                        yj = linkstackk[0][1] + neigb[j][1]
                        zj = linkstackk[0][2] + neigb[j][2]
                        if skeilvol.readValue((xj, yj, zj)):
                            count = count + 1
                            if not visited.readValue((xj, yj, zj)):
                                linkstackk.append([xj, yj, zj])
                                neigb_skeil.append([xj, yj, zj, branchn])

                    visited.addValue(
                        (linkstackk[0][0], linkstackk[0][1], linkstackk[0][2]),
                        1)

                    if count < 3: # Point belongs to current branch
                        member.append((linkstackk[0]))
                    else:
                        branch_dictn['end'] = (linkstackk[0])
                        startnode.extend(
                            neigb_skeil)
                        for k in range(0, len(neigb_skeil)):
                            visited.addValue(
                                (neigb_skeil[k][0], neigb_skeil[k][1],
                                 neigb_skeil[k][2]), 1)
                        break

                    del linkstackk[0]
                branch_dictn['member'] = copy.deepcopy(member)
                branch_dictn['fatherindex'] = startnode[0][3]
                branch_dict.append(branch_dictn)

                del linkstack[l - flag]
                flag = flag + 1
            branch_dictn = {}
            branchn = branchn + 1
            branch_dictn['index'] = branchn
            branch_dictn['start'] = [
                startnode[0][0], startnode[0][1], startnode[0][2]
            ]
            if debug != 0:
                member = []
        while linkstack:
            count = 0
            neigb_skeil = []
            for j in range(0, 26):
                xj = linkstack[0][0] + neigb[j][0]
                yj = linkstack[0][1] + neigb[j][1]
                zj = linkstack[0][2] + neigb[j][2]
                if skeilvol.readValue((xj, yj, zj)):
                    count = count + 1
                    if not visited.readValue((xj, yj, zj)):
                        linkstack.append([xj, yj, zj])
                        neigb_skeil.append([xj, yj, zj, branchn])

            visited.addValue(
                (linkstack[0][0], linkstack[0][1], linkstack[0][2]), 1)

            if count < 3:
                member.append((linkstack[0]))
            else:
                branch_dictn['end'] = (linkstack[0])
                startnode.extend(neigb_skeil)
                for k in range(0, len(neigb_skeil)):
                    visited.addValue((neigb_skeil[k][0], neigb_skeil[k][1],
                                      neigb_skeil[k][2]), 1)
                break

            del linkstack[0]

        branch_dictn['member'] = copy.deepcopy(member)
        branch_dictn['fatherindex'] = startnode[0][3]
        branch_dict.append(branch_dictn)
        del startnode[0]

    return branch_dict

def compute_base_vector(LABEL_TRANS, order):
    minz = np.min(np.where(LABEL_TRANS == 1)[2])
    maxz = np.max(np.where(LABEL_TRANS == 1)[2])
    cha = maxz - minz
    if order == 1:
        center1_z = int(maxz - 0.1 * cha)
        zzz2 = 0.6
        center2_z = int(zzz2 * cha + minz)
    else:
        center1_z = int(minz + 0.1 * cha)
        zzz2 = 0.4
        center2_z = int(zzz2 * cha + minz)
    image1 = LABEL_TRANS[:, :, center1_z]
    image2 = LABEL_TRANS[:, :, center2_z]
    center1_x = np.mean(np.argwhere(image1 > 0)[:, 0])
    center1_y = np.mean(np.argwhere(image1 > 0)[:, 1])
    center2_x = np.mean(np.argwhere(image2 > 0)[:, 0])
    center2_y = np.mean(np.argwhere(image2 > 0)[:, 1])

    if order == 1:
        basev = np.array([
            center2_x - center1_x, center2_y - center1_y,
            center1_z - center2_z
        ])
    else:
        basev = np.array([
            center2_x - center1_x, center2_y - center1_y,
            center2_z - center1_z
        ])

    return basev

def cosine(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    cosine_sim = dot_product / (norm_a * norm_b)
    return cosine_sim

def find_mainpart_index(maxzzz,Bi, basev):
    """
    This function processes the 'Bi' structure and finds the index of the main part
    based on cosine similarity with the provided base vector.
    
    Args:
    - Bi: List of airway branch information, where each element contains a 'member' and 'start'.
    - basev: A numpy array representing the base vector for cosine similarity calculation.
    
    Returns:
    - mmm: The index of the main part in 'Bi', or None if no main part is found.
    """
    mainpart = []

    # Collect cosine similarities for branches with more than 20 members
    for i, bi in enumerate(Bi):
        if i > 20:
            break
        if len(bi['member']) == 0:
            continue
        if len(bi['member']) >maxzzz/3.6: 
            break
        v = np.array(bi['member'][-1]) - np.array(bi['start'])

        if len(bi['member']) > 12:
            mainpart.append([i, cosine(basev, v),len(bi['member'])])

    flag = 0
    mmm = 0

    # Find the index of the main part based on cosine similarity
    for index,(i, cos,lll) in enumerate(mainpart):
        if cos < 0.928 and flag == 0:
            continue
        if cos > 0.928:
            flag = 1
        if cos < 0.93 and flag == 1:
            mmm = i
            break

    return mmm

def smooth_points(ori_ps):
    inter = 3
    selected_indices = np.arange(0, len(ori_ps), len(ori_ps) // inter)
    selected_indices = np.append(selected_indices, [len(ori_ps) - 1], axis=0)
    if abs(selected_indices[-2]-selected_indices[-1])<5:
        selected_indices = np.delete(selected_indices, -2)

    selected_points = ori_ps[selected_indices, :] 

    x_interp = interp1d(selected_indices,
                        selected_points[:, 0],
                        kind='linear',
                        fill_value="extrapolate")
    y_interp = interp1d(selected_indices,
                        selected_points[:, 1],
                        kind='linear',
                        fill_value="extrapolate")
    z_interp = interp1d(selected_indices,
                        selected_points[:, 2],
                        kind='linear',
                        fill_value="extrapolate")

    interpolated_points = np.array([
        x_interp(np.linspace(0,
                             len(ori_ps) - 1, len(ori_ps))),
        y_interp(np.linspace(0,
                             len(ori_ps) - 1, len(ori_ps))),
        z_interp(np.linspace(0,
                             len(ori_ps) - 1, len(ori_ps)))
    ]).T

    new_a = np.zeros_like(interpolated_points, dtype=int)

    new_a[0] = np.round(interpolated_points[0]).astype(int)

    for i in range(1, len(interpolated_points)):
        x, y, z = np.round(interpolated_points[i]).astype(int)

        if abs(x - new_a[i - 1][0]) > 1:
            x = new_a[i - 1][0] + np.sign(x - new_a[i - 1][0])
        if abs(y - new_a[i - 1][1]) > 1:
            y = new_a[i - 1][1] + np.sign(y - new_a[i - 1][1])
        if abs(z - new_a[i - 1][2]) > 1:
            z = new_a[i - 1][2] + np.sign(z - new_a[i - 1][2])

        new_a[i] = [x, y, z]

    indices = np.argsort(new_a[:, 2])
    new_a = new_a[indices, :]


    # Step 1: Remove duplicates based on the third column (z)
    unique_z_points = []
    last_z = None
    for point in new_a:
        if point[2] != last_z:
            unique_z_points.append(point)
            last_z = point[2]

    unique_z_points = list(reversed(unique_z_points))

    # Step 2: Ensure continuity between adjacent points
    final_points = [unique_z_points[0]]  # Start with the first point

    for i in range(1, len(unique_z_points)):
        x, y, z = unique_z_points[i]

        # Ensure continuity in x, y, and z within ±1 range from the last point
        prev_x, prev_y, prev_z = final_points[-1]

        # Adjust to ensure the difference is within ±1
        if abs(x - prev_x) > 1:
            x = prev_x + np.sign(x - prev_x)
        if abs(y - prev_y) > 1:
            y = prev_y + np.sign(y - prev_y)
        if abs(z - prev_z) > 1:
            z = prev_z + np.sign(z - prev_z)

        final_points.append([x, y, z])

    final_points = np.array(final_points)
    flipped_points = np.flip(final_points, axis=0)

    return flipped_points


def process_mainairway_points(B, Bi, mmm):
    """
    This function processes the main airway based on input data, modifies airway structure, 
    and returns the updated airway points (B) after smoothing and filtering.

    Args:
    - B: The current airway structure represented as a 2D numpy array.
    - Bi: The airway branch information.
    - mmm: The index of the main airway in Bi (used for iteration limit).

    Returns:
    - B: The updated airway structure after processing.
    """
    mainairway = []
    for i, bi in enumerate(Bi):
        if i >= mmm:
            break
        mainairway.append(bi['start'])
        mainairway += bi['member']
        if 'end' in bi:
            mainairway.append(bi['end'])
    mainairway = np.array(mainairway)
    mainairway = np.unique(mainairway, axis=0)


    fullyB_reversed = np.array([row for row in B[::-1]])
    index_map = {tuple(row): i for i, row in enumerate(fullyB_reversed)}

    def sort_key(row):
        """Sort key based on the reversed index map."""
        return index_map[tuple(row)]

    mainairway = np.array(sorted(mainairway, key=sort_key))


    newmain = smooth_points(mainairway)

    cut_main = mainairway[:len(mainairway) - len(newmain)]
    set_b = set(map(tuple, cut_main))
    result = [row.tolist() for row in B if tuple(row) not in set_b]
    B = np.array(result)
    mainairway = mainairway[len(mainairway) - len(newmain):] 

    rows_to_replace = [
        np.where(np.all(B == m, axis=1))[0] for m in reversed(mainairway)
    ]
    rows_to_replace = [item for sublist in rows_to_replace for item in sublist]

    i = len(newmain) - 1
    for index in rows_to_replace:
        B[index, :] = newmain[i]
        i -= 1

    return B

def merging(branch_dict, len_thre):
    """
    Merges branches from a branch dictionary based on the threshold length and the relationship 
    between parent and child branches. The function operates in two main phases: 
    - It first identifies and merges short branches with their parent branches.
    - Then it handles branches that are single descendants of a parent, merging them accordingly.

    Args:
        branch_dict (list): A list of dictionaries representing branches. Each dictionary contains 
                             information about a branch, such as 'start', 'end', 'member', and 'fatherindex'.
        len_thre (int): A threshold length used to determine whether a branch should be merged. 
                         Branches shorter than or equal to this length may be merged with their parent.

    Returns:
        list: The updated branch dictionary after merging operations.
    """
    cut_list = []
    for i in range(len(branch_dict)):
        bi = branch_dict[i]['member'].copy()
        bi.insert(0, branch_dict[i]['start'])
        if 'end' in branch_dict[i]:
            bi.append(branch_dict[i]['end'])
        if len(bi) <= len_thre:
            sons = 0
            for j in range(i + 1, len(branch_dict)):
                fj = branch_dict[j]['fatherindex']
                if fj == i + 1:
                    sons += 1
                    if sons == 1:
                        cut_list.append(i)
                        branch_dict[j]['fatherindex'] = branch_dict[i][
                            'fatherindex']
                        cuti = branch_dict[i]['member'].copy()
                        if 'end' in branch_dict[i]:
                            cuti.append(branch_dict[i]['end'])
                        cuti.append(branch_dict[j]['start'])
                        branch_dict[j]['start'] = branch_dict[i]['start'].copy(
                        )
                        branch_dict[j][
                            'member'] = cuti + branch_dict[j]['member']
                    if sons > 1:
                        branch_dict[j]['fatherindex'] = branch_dict[i][
                            'fatherindex']
                        cuti = branch_dict[i]['member'].copy()
                        if 'end' in branch_dict[i]:
                            cuti.append(branch_dict[i]['end'])
                        cuti.append(branch_dict[j]['start'])
                        branch_dict[j]['start'] = branch_dict[i]['start'].copy(
                        )
                        branch_dict[j][
                            'member'] = cuti + branch_dict[j]['member']
            if sons == 0:
                cut_list.append(i)
    new_list = [x for i, x in enumerate(branch_dict) if i not in cut_list]
    branch_dict = new_list

    cut_s = []
    child_num = np.zeros(branch_dict[-1]['index'], dtype=int)
    for i in range(len(branch_dict)):
        f = branch_dict[i]['fatherindex']
        child_num[f] += 1
    single = np.where(child_num == 1)
    single = list(single[0])[1:]
    single_index = []
    for s in single:
        for i in range(len(branch_dict)):
            if branch_dict[i]['index'] == s:
                single_index.append(i)
    fs_index = np.zeros((len(single_index), 2), dtype=int)
    for s in range(len(single_index) - 1, -1, -1):
        for i in range(len(branch_dict) - 1, -1, -1):
            if branch_dict[i]['fatherindex'] == branch_dict[
                    single_index[s]]['index']:
                fs_index[s, 0] = branch_dict[i]['fatherindex']
                fs_index[s, 1] = branch_dict[i]['index']
                cut_s.append(i)
                bi = branch_dict[i]['member'].copy()
                bi.insert(0, branch_dict[i]['start'])
                bi.insert(0, branch_dict[single_index[s]]['end'])
                if 'end' in branch_dict[i]:
                    branch_dict[single_index[s]]['end'] = branch_dict[i]['end']
                else:
                    branch_dict[single_index[s]]['end'] = bi[-1]
                    bi = bi[:-1]
                branch_dict[single_index[s]]['member'] = branch_dict[
                    single_index[s]]['member'] + bi
    for s in range(len(fs_index) - 1, -1, -1):
        for i in range(len(branch_dict) - 1, -1, -1):
            if branch_dict[i]['fatherindex'] == fs_index[s, 1]:
                branch_dict[i]['fatherindex'] = fs_index[s, 0]
    new_list = [x for i, x in enumerate(branch_dict) if i not in cut_s]
    branch_dict = new_list

    return branch_dict

def remerging(branch_dict,branch_dict_g,remerge_l): 
    #大
    cut_l=np.zeros(len(remerge_l),dtype=int)+1000
    t=np.zeros(len(remerge_l),dtype=int)+1000
    flag=np.zeros(len(remerge_l),dtype=int)
    for i in range(len(branch_dict)):
        if branch_dict_g[i]['fatherindex'] in remerge_l:
            flag[remerge_l.index(branch_dict_g[i]['fatherindex'])]+=1
            bi = branch_dict[i]['member'].copy()
            bi.insert(0, branch_dict[i]['start'])
            if 'end' in branch_dict[i]:
                bi.append(branch_dict[i]['end'])
            if len(bi)<=t[remerge_l.index(branch_dict_g[i]['fatherindex'])]:
                t[remerge_l.index(branch_dict_g[i]['fatherindex'])]=len(bi)
                cut_l[remerge_l.index(branch_dict_g[i]['fatherindex'])]=i
    cut_l=list(cut_l)
    br3=list(np.where(flag>2)[0])
    cut_l=[n for i, n in enumerate(cut_l) if i not in br3]
    for i in cut_l:
        for j in range(i+1,len(branch_dict)):
            if branch_dict[j]['fatherindex']==branch_dict[i]['index']:
                branch_dict[j]['fatherindex']=branch_dict[i]['fatherindex']
                cuti = branch_dict[i]['member'].copy()
                if 'end' in branch_dict[i]:
                    cuti.append(branch_dict[i]['end'])
                cuti.append(branch_dict[j]['start'])
                branch_dict[j]['start']=branch_dict[i]['start'].copy()
                branch_dict[j]['member']= cuti+branch_dict[j]['member']
    new_list = [x for i, x in enumerate(branch_dict) if i not in cut_l]
    branch_dict=new_list
    return branch_dict

def tree_parsing_func(skeleton_parse, label, cd):
    #parse the airway tree
    edt, inds = ndimage.distance_transform_edt(1-skeleton_parse, return_indices=True)
    tree_parsing = np.zeros(label.shape, dtype = np.uint16)
    tree_parsing = cd[inds[0,...], inds[1,...], inds[2,...]] * label
    return tree_parsing

class Topology_Tree:
    """
    Class for processing a 3D skeleton representation of a tree-like structure.
    It handles tasks like branch extraction, tree topology analysis, and calculating properties such as branch size and position.

    Attributes:
        label: A 3D binary array representing the segmented tree structure.
        order: Defines the order of the z-axis in the tree's skeletal structure.
               If order == 1, the z-axis is inverted in the tree structure.
               If order == 0 (default), no inversion is applied.
        B: List of coordinates representing the 3D skeleton.
        Bi: List of branches derived from the skeleton.
        o: Coordinates of the tree's origin (mean position of the skeleton points).
        psize: Scaling factors for resizing the skeleton's dimensions.
        merge_t: Threshold used for merging branches in the tree.
        colors: List of colors for visualization.
    """

    def __init__(self, LA, order, merge_t,remerge_l=[]):
        """
        Initializes the Topology_Tree instance with the given parameters.
        
        Parameters:
            LA: 3D binary array representing the segmented tree structure.
            order: Defines the order of the z-axis in the tree's skeletal structure (0 or 1).
            merge_t: Threshold for merging branches in the tree.
        """
        self.label = LA
        self.order = order
        self.B = []
        self.Bi = []
        self.o = []
        self.psize = []
        self.B_resize = []
        self.Bi_resize = []
        self.merge_t = merge_t
        self.remerge_l=remerge_l
        self.colors = []
        self.rb23=0
        self.rb12=0
        self.rb45=0
        self.rb6=0
        self.lb123=0
        self.l010=0
        self.numofzs=0
        self.rb123=0

    def sub(self):
        """
        Processes the skeletal structure of the tree, including:

        Refines the skeleton with hole-filling, dilation, closing, and skeletonization.
        Analyzes and smooths the centerline of the trachea if necessary, updating the skeleton.
        Extracts branches and updates them after centerline smoothing.
        """
        LABEL_TRANS = binary_fill_holes(binary_dilation(self.label))
        LABEL_TRANS = binary_closing(LABEL_TRANS)
        LABEL_TRANS = maximum_3d(LABEL_TRANS)
        skel = skeletonize_3d(LABEL_TRANS)
        B = np.array(np.where(skel != 0))
        B = B[:, B[2].argsort()]
        B = B.T
        ox = np.mean(B[:, 0])
        oy = np.mean(B[:, 1])
        oz = np.mean(B[:, 2])
        self.o = [ox, oy, oz]
        if self.order == 1:
            B[:, 2] = self.label.shape[2] - B[:, 2]
        Bi = subsection(B, debug=1)
        basev = compute_base_vector(LABEL_TRANS, self.order)
        mmm = find_mainpart_index(B[0,2],Bi, basev)
        if mmm > 1:
            B = process_mainairway_points(B, Bi, mmm)
            newBi = subsection(B, debug=1)
            self.B = B
            self.Bi = newBi
        else:
            self.B = B
            self.Bi = Bi

    def merge(self):
        """
        Merges branches in the tree based on the given threshold.
        
        The method processes the branches stored in `Bi` and merges them if their 
        distance is smaller than the threshold defined in `merge_t`. 
        """
        Bi = merging(self.Bi, self.merge_t)
        if self.order == 1:
            for i in range(Bi.__len__()):
                Bi[i]['start'][2] = self.label.shape[2] - Bi[i]['start'][2]
                if 'end' in Bi[i].keys():
                    Bi[i]['end'][2] = self.label.shape[2] - Bi[i]['end'][2]
                if Bi[i]['member'] != []:
                    member = np.array(Bi[i]['member'])
                    member[:, 2] = self.label.shape[2] - member[:, 2]
                    Bi[i]['member'] = member.tolist()
        self.Bi = Bi

    def grade(self):
        Bi_g = copy.deepcopy(self.Bi)
        flag = np.zeros(self.Bi.__len__(), dtype=np.int8)
        grade = '0'
        Bi_g[0]['index'] = grade
        Bi_g[0]['fatherindex'] = '-1'
        if self.Bi[1]['start'][1] > self.Bi[2]['start'][1]:
            Bi_g[1]['index'] = '01'
            Bi_g[1]['fatherindex'] = '0'
            Bi_g[2]['index'] = '00'
            Bi_g[2]['fatherindex'] = '0'
        else:
            Bi_g[1]['index'] = '00'
            Bi_g[1]['fatherindex'] = '0'
            Bi_g[2]['index'] = '01'
            Bi_g[2]['fatherindex'] = '0'
        for i in range(3, self.Bi.__len__()):
            for g in range(len(self.Bi)):
                if self.Bi[g]['index'] == self.Bi[i]['fatherindex']:
                    string = Bi_g[g]['index'] + str(flag[g])
                    break
            flag[g] += 1
            Bi_g[i]['index'] = string
            Bi_g[i]['fatherindex'] = Bi_g[g]['index']

        self.Bi_g = Bi_g

    def remerge(self):
        Bi=remerging(self.Bi,self.Bi_g,self.remerge_l)
        self.Bi=Bi
        Topology_Tree.grade(self)

    def regrade(self):
        
        vectors = [np.array([0, -1, 0]),np.array([0, 1, 0])]
        self._process_grade('0',vectors, self._main_left_right)
        
        if self.order==1:
            vectors = [np.array([0, -1, 0.1]),np.array([0, -1, -1])]
        else:
            vectors = [np.array([0, -1, 0.1]),np.array([0, -1, 1])]
        self._process_grade('00', vectors,self._right)
        
        if self.order==1:
            vectors = [np.array([0, 0, 1]), np.array([-1, -1, 0]),np.array([1, 0, 0])] 
        else:
            vectors = [np.array([0, 0, -1]), np.array([-1, -1, 0]),np.array([1, 0, 0])]
        self._process_grade('000', vectors,self._right_upper)
        
        if self.order==1:
            vectors = [np.array([1, -1, -0.25]), np.array([0, 0, -1])] 
        else:
            vectors = [np.array([1, -1, 0.25]),  np.array([0, 0, 1])]
        self._process_grade('001', vectors,self._right_middle)
        
        vectors = [np.array([0, -1, 0]),np.array([1, 0, 0])]
        self._process_grade('0010', vectors,self._seg0010)
        
        vectors = [np.array([0, 1, 0]),np.array([0, -1, 0])]
        self._process_grade('00111', vectors,self._seg00111)
        
        vectors = [np.array([0, -1,0]),np.array([0, 1, 0])]
        self._process_grade('001111', vectors,self._seg001111)
        
        vectors = [np.array([0, -1,0]),np.array([0, 1, 0])]
        self._process_grade('0011111', vectors,self._seg0011111)

        
        if self.order==1:
            vectors = [np.array([0, 1, 0]),np.array([0, 0.18, -1])]
        else:
            vectors = [np.array([0, 1, 0]),np.array([0, 0.18, 1])]
        self._process_grade('01', vectors,self._left)
        
        if self.order==1:
            vectors = [np.array([0, 0, 1]), np.array([0, 0, -1])]
        else:
            vectors = [np.array([0, 0, -1]),  np.array([0, 0, 1])]
        self._process_grade('010', vectors,self._left_upper)
        
        if self.order==1:
            vectors = [np.array([0, 1, 0]),np.array([1, 0, -1])]
        else:
            vectors = [np.array([0, 1, 0]),np.array([1, 0, 1])]
        self._process_grade('0101', vectors,self._seg0101)
        
        if self.order==1:
            vectors = [np.array([-1, 0, 0]), np.array([0, 0, -1])]
        else:
            vectors = [np.array([-1, 0, 0]),  np.array([0, 0, 1])]
        self._process_grade('011', vectors,self._seg011)
        
        if self.order==1:
            vectors= [np.array([1, 1, 0]),np.array([0,0 , -1])]
        else:
            vectors = [np.array([1, 1, 0]),np.array([0,0 , 1])]
        self._process_grade('0111', vectors,self._seg0111)
        
        vectors= [np.array([0, 1, 0]),np.array([0, -1, 0])]
        self._process_grade('01111', vectors,self._seg01111)

    def _process_grade(self, startgrade, vectors,grade_func):
        segments = [seg.copy() for seg in self.Bi_g if seg['fatherindex'] == startgrade]
        segments = sorted(segments, key=lambda x: x['index'])
        if len(segments) > 1:
            grade_func(startgrade, vectors,segments)

    def _main_left_right(self, startgrade,vectors, Blr):
        Blr_values = np.array([self._calculate_similarity(Blr, vector) for vector in vectors])
        haoma = ['00', '01'] 
        if Blr_values.shape[1] == 2 :
            self._update_segment_codes( Blr,Blr_values,haoma)

    def _left(self, startgrade,vectors, Budr):
        Bud_values = np.array([self._calculate_similarity(Budr, vector) for vector in vectors])
        haoma = ['010', '011'] 
        if max(Bud_values[0,:])<=0.7 or max(Bud_values[:,0])<=0.7:
            self.l010 = 1 
            self._handle_missing_branch(startgrade)
        self._update_segment_codes( Budr,Bud_values,haoma)

    def _left_upper(self, startgrade,vectors, B12345r):
        B12345_values = np.array([self._calculate_similarity(B12345r, vector) for vector in vectors])
        haoma = ['0100', '0101'] 
        if B12345_values.shape[1] ==2:
            if max(B12345_values[0,:])<=0.4:
                self.lb123 = 1 
                self._handle_missing_branch(startgrade)
            self._update_segment_codes( B12345r,B12345_values,haoma)

            if self.order==1:
                vectors = [np.array([-1, 0, 1]), np.array([1, 0, 0])]
            else:
                vectors = [np.array([-1, 0, -1]),  np.array([1, 0, 0])]
            self._process_grade('0100', vectors,self._seg0100)
        elif B12345_values.shape[1] ==3:
            if self.order==1:
                vectors = [np.array([-1, 0, 1]), np.array([1, 0, 0]), np.array([0, 0, -1])]
            else:
                vectors = [np.array([-1, 0, -1]),np.array([1, 0, 0]),  np.array([0, 0, 1])]
            B12345_values = np.array([self._calculate_similarity(B12345r, vector) for vector in vectors])
            haoma=['01000','01001','0101']
            self._update_segment_codes( B12345r,B12345_values,haoma)

    def _seg0100(self, startgrade,vectors, B123r):
        B123_values = np.array([self._calculate_similarity(B123r, vector) for vector in vectors])
        haoma = ['01000', '01001'] 
        if B123_values.shape[1] == 2 :
            self._update_segment_codes( B123r,B123_values,haoma)
        elif B123_values.shape[1] ==3:
            if self.order==1:
                vectors = [np.array([-1, 0, 1]), np.array([0, 1, -0.1]),np.array([1, 0, 0])]
            else:
                vectors = [np.array([-1, 0, -1]), np.array([0, 1, 0.1]),np.array([1, 0, 0])]
            B123_values = np.array([self._calculate_similarity(B123r, vector) for vector in vectors])
            haoma=['01000','01001','01002']
            self._update_segment_codes( B123r,B123_values,haoma)

    def _seg0101(self, startgrade,vectors, B45r):
        B45_values = np.array([self._calculate_similarity(B45r, vector) for vector in vectors])
        haoma = ['01010', '01011'] 
        if B45_values.shape[1] ==2:
            self._update_segment_codes( B45r,B45_values,haoma)

    def _seg011(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['0110', '0111'] 
        if B610_values.shape[1] ==2:
            self._update_segment_codes( B610r,B610_values,haoma)

    def _seg0111(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['01110', '01111'] 
        if B610_values.shape[1] == 2 :
            self._update_segment_codes( B610r,B610_values,haoma)
        elif B610_values.shape[1] ==3:
            if self.order==1:
                vectors= [np.array([1, 1, 0]),np.array([0,0.3 , -1]),np.array([0,-0.3 , -1])]
            else:
                vectors = [np.array([1, 1, 0]),np.array([0,0.3 , 1]),np.array([0,-0.3 , 1])]
            B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
            haoma=['01110','011110','011111']
            self._update_segment_codes( B610r,B610_values,haoma)

    def _seg01111(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['011110', '011111'] 
        if B610_values.shape[1] ==2:
            self._update_segment_codes( B610r,B610_values,haoma)

    def _right(self, startgrade,vectors, Budr):
        Bud_values = np.array([self._calculate_similarity(Budr, vector) for vector in vectors])
        haoma = ['000', '001'] 
        if Bud_values.shape[1] > 2 and np.where(np.max(Bud_values, axis=0) <= 0.85)[0].size == 1:##
            self._handle_multiple_branches(haoma, Budr, Bud_values, startgrade,vectors)
        elif max(Bud_values[0, :]) <= 0.85:
            self.rb123 = 1 
            self._handle_missing_branch(startgrade)
        elif Bud_values.shape[1] ==2:
            self._update_segment_codes( Budr,Bud_values,haoma)

    def _right_upper(self, startgrade,vectors, B123r):
        B123_values = np.array([self._calculate_similarity(B123r, vector) for vector in vectors])
        haoma=['0000','0001','0002']
        if B123_values.shape[1] ==3 :
            self._update_segment_codes(B123r,B123_values,haoma)

    def _right_middle(self, startgrade,vectors, B45r):
        B45_values = np.array([self._calculate_similarity(B45r, vector) for vector in vectors])
        haoma = ['0010', '0011'] 
        if B45_values.shape[1] ==2:
            if max(B45_values[0, :]) <= 0.5: 
                self.rb45 = 1 
                self._handle_missing_branch(startgrade)
            self._update_segment_codes( B45r,B45_values,haoma)

            if self.order==1:
                vectors = [np.array([-1, -0.1, 0]), np.array([0, 0, -1])]
            else:
                vectors = [np.array([-1, -0.1, 0]),  np.array([0, 0, 1])]
            self._process_grade('0011', vectors,self._seg0011)
        elif B45_values.shape[1] ==3:
            if self.order==1:
                vectors = [np.array([1, -0.7, 0]), np.array([-1, 0, 0]), np.array([0, -0.4, -1])]
            else:
                vectors = [np.array([1, -0.7, 0]), np.array([-1, 0, 0]),  np.array([0, -0.4, 1])]
            B45_values = np.array([self._calculate_similarity(B45r, vector) for vector in vectors])
            haoma=['0010','00110','00111']
            self._update_segment_codes( B45r,B45_values,haoma)

    def _seg0011(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['00110', '00111'] 
        if B610_values.shape[1] ==2:
            if max(B610_values[0, :]) <= 0.5:
                self.rb6 = 1
                self._handle_missing_branch(startgrade)
            self._update_segment_codes( B610r,B610_values,haoma)

    def _seg0010(self, startgrade,vectors, B45r):
        B45_values = np.array([self._calculate_similarity(B45r, vector) for vector in vectors])
        haoma = ['00100', '00101'] 
        if B45_values.shape[1] == 2 :
            self._update_segment_codes( B45r,B45_values,haoma)

    def _seg00111(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['001110', '001111'] 
        if B610_values.shape[1] == 2 :
            self._update_segment_codes( B610r,B610_values,haoma)
        elif B610_values.shape[1] ==3:
            if self.order==1:
                vectors= [np.array([0, -1, 0]),np.array([0,-0.1 , -1]),np.array([0,0.3 , -1])]
            else:
                vectors = [np.array([0, -1, 0]),np.array([0,-0.1, 1]),np.array([0,0.3 , 1])]
            B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
            haoma=['0011110','0011111','001110']
            self._update_segment_codes( B610r,B610_values,haoma)

    def _seg001111(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['0011110', '0011111'] 
        if B610_values.shape[1] == 2 :
            self._update_segment_codes( B610r,B610_values,haoma)
        elif B610_values.shape[1] ==3:
            if self.order==1:
                vectors= [np.array([0, -1, 0]),np.array([0,-0.4 , -1]),np.array([0,0.2 , -1])]
            else:
                vectors = [np.array([0, -1, 0]),np.array([0,-0.4, 1]),np.array([0,0.2 , 1])]
            B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
            haoma=['0011110','00111110','00111111']
            self._update_segment_codes( B610r,B610_values,haoma)

    def _seg0011111(self, startgrade,vectors, B610r):
        B610_values = np.array([self._calculate_similarity(B610r, vector) for vector in vectors])
        haoma = ['00111110', '00111111'] 
        if B610_values.shape[1] == 2 :
            self._update_segment_codes( B610r,B610_values,haoma)

    def _calculate_similarity(self,segments,vector):
        similarities = []
        for seg in segments:
            start_point = np.array(seg['start'])
            end_point = np.array(seg['end'] if 'end' in seg else seg['member'][-1])
            similarities.append(cosine(end_point-start_point, vector))
        return similarities
    
    def _handle_multiple_branches(self, haoma, segments, segment_values, startgrade,vectors):
        viewed = []
        wrongb = list(np.where(np.max(segment_values, axis=0) <= 0.75)[0])##33
        for i, seg in enumerate(segments):
            if i in wrongb:
                continue
            newgrade = haoma[0]
            haoma.pop(0)
            if newgrade == seg['index']:
                continue
            for j in range(len(self.Bi_g)):
                if self.Bi_g[j]['index'][:len(seg['index'])] == seg['index'] and self.Bi_g[j]['index'] != seg['index'] and j not in viewed:
                    viewed.append(j)
                    numj = self.Bi_g[j]['index'][len(seg['index']):]
                    self.Bi_g[j]['index'] = newgrade + numj
                    numfj = self.Bi_g[j]['fatherindex'][len(seg['index']):]
                    self.Bi_g[j]['fatherindex'] = newgrade + numfj
            seg['index'] = newgrade
        segments = [seg.copy() for seg in self.Bi_g if seg['fatherindex'] == startgrade]
        segments = sorted(segments, key=lambda x: x['index'])
        segment_values = np.array([self._calculate_similarity(segments, vector) for vector in vectors])
        segment_values = np.delete(segment_values, wrongb, axis=1)
        if np.argmax(segment_values[:, 0]) != 0 and np.argmax(segment_values[:, 1]) != 1:
            self._exchange_grade(startgrade, segments)

    def _handle_missing_branch(self, startgrade):
        for j in range(len(self.Bi_g)):
            if self.Bi_g[j]['index'][:len(startgrade)] == startgrade and self.Bi_g[j]['index'] != startgrade:
                self.Bi_g[j]['index'] = startgrade + str(1) + self.Bi_g[j]['index'][len(startgrade):]
                self.Bi_g[j]['fatherindex'] = startgrade + str(1) + self.Bi_g[j]['fatherindex'][len(startgrade):]
    
    def _update_segment_codes(self,bro, similarity_values, haoma):
        viewed = []
        # Initialize grades and assignment states
        new_grades = [None] * len(bro)
        assigned = [False] * len(haoma)
        used_codes = set()
        remaining_grades = list(range(len(bro)))

        # Assign codes dynamically
        while remaining_grades:
            # Sort remaining segments by max similarity
            remaining_grades.sort(key=lambda i: -max(similarity_values[:, i]))
            current = remaining_grades.pop(0)

            # Assign the most suitable code
            sorted_indices = np.argsort(-similarity_values[:, current])
            for idx in sorted_indices:
                candidate_code = haoma[idx]
                if not assigned[idx] or len(remaining_grades) == len(haoma) - len(used_codes):
                    new_grades[current] = candidate_code
                    assigned[idx] = True
                    used_codes.add(candidate_code)
                    break

        # Update segment codes in Bi_g
        for i, seg in enumerate(bro):
            newgrade = new_grades[i]
            if newgrade == seg['index']:
                continue

            for j in range(len(self.Bi_g)):
                if self.Bi_g[j]['index'][:len(seg['index'])]==seg['index'] and self.Bi_g[j]['index']!=seg['index'] and j not in viewed: ###### B8 9 10    and j not in viewed
                    viewed.append(j)                        
                    numj=self.Bi_g[j]['index'][len(seg['index']):]
                    self.Bi_g[j]['index']=newgrade+numj
                    numfj=self.Bi_g[j]['fatherindex'][len(seg['index']):]
                    self.Bi_g[j]['fatherindex']=newgrade+numfj
                if self.Bi_g[j]['index']==seg['index'] and j not in viewed:
                    self.Bi_g[j]['index']=newgrade  
                    viewed.append(j)

    def resize(self, px, py, pz,dir=None):
        """
        Resizes the branches and skeleton coordinates based on the given scaling factors.

        Parameters:
            px: Scaling factor for the x-axis.
            py: Scaling factor for the y-axis.
            pz: Scaling factor for the z-axis.
        
        This method rescales the coordinates of the skeleton and branches according 
        to the scaling factors, while also adjusting for the tree's origin.
        """
        self.Bi_resize = copy.deepcopy(self.Bi)
        self.B_resize = copy.deepcopy(self.B)
        self.psize = [px, py, pz]
        self.B_resize = self.B_resize.astype(np.float32)
        self.B_resize[:, 0] = (self.B[:, 0] - self.o[0]) * self.psize[0]
        self.B_resize[:, 1] = (self.B[:, 1] - self.o[1]) * self.psize[1]
        self.B_resize[:, 2] = (self.B[:, 2] - self.o[2]) * self.psize[2]
        for i in range(self.Bi_resize.__len__()):
            self.Bi_resize[i]['start'][0] = (self.Bi_resize[i]['start'][0] -
                                             self.o[0]) * self.psize[0]
            self.Bi_resize[i]['start'][1] = (self.Bi_resize[i]['start'][1] -
                                             self.o[1]) * self.psize[1]
            self.Bi_resize[i]['start'][2] = (self.Bi_resize[i]['start'][2] -
                                             self.o[2]) * self.psize[2]
            if 'end' in self.Bi_resize[i].keys():
                self.Bi_resize[i]['end'][0] = (self.Bi_resize[i]['end'][0] -
                                               self.o[0]) * self.psize[0]
                self.Bi_resize[i]['end'][1] = (self.Bi_resize[i]['end'][1] -
                                               self.o[1]) * self.psize[1]
                self.Bi_resize[i]['end'][2] = (self.Bi_resize[i]['end'][2] -
                                               self.o[2]) * self.psize[2]
            if self.Bi_resize[i]['member'] != []:
                member = np.array(self.Bi_resize[i]['member'])
                member = member.astype(np.float32)
                member[:, 0] = (member[:, 0] - self.o[0]) * self.psize[0]
                member[:, 1] = (member[:, 1] - self.o[1]) * self.psize[1]
                member[:, 2] = (member[:, 2] - self.o[2]) * self.psize[2]
                self.Bi_resize[i]['member'] = member.tolist()
        if dir:
            np.save(dir, np.array(self.Bi_resize))
    
    def recons(self, dir):
        """
        Reconstructs the airway tree surface using marching cubes algorithm.

        Parameters:
            dir: The directory where the reconstructed mesh will be saved.
        
        This method uses the marching cubes algorithm to generate a 3D mesh 
        representing the airway tree's surface and saves it in the specified directory.
        """
        iso = 0.95
        verts, faces, _, _ = marching_cubes(self.label, iso)
        verts[:, 0] = verts[:, 0] - self.o[0]
        verts[:, 1] = verts[:, 1] - self.o[1]
        verts[:, 2] = verts[:, 2] - self.o[2]
        verts[:, 0] = verts[:, 0] * self.psize[0]
        verts[:, 1] = verts[:, 1] * self.psize[1]
        verts[:, 2] = verts[:, 2] * self.psize[2]
        faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
        mesh_airway = pv.PolyData(verts, faces)
        mesh_airway_smooth = mesh_airway.smooth(relaxation_factor=0.2)
        pl = pv.Plotter()
        pl.add_mesh(mesh_airway_smooth, color='#E96C6F', style='surface')
        mesh_airway_smooth.save(dir)
    def sub_model(self, st, save_dir, case):
        """
        Visualizes and saves a 3D model of the airway tree with branches.

        Parameters:
            st: The start time to compute the duration of airway tree parsing.
            save_dir: Directory to save the output visualization and animation.
            case: The case identifier used to load the relevant files.
        
        This method generates a 3D visualization of the airway tree with its branches 
        and saves the result as an STL file and GIF animation.
        """
        if self.colors == []:
            color_list = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
                'C', 'D', 'E', 'F'
            ]
            for i in range(400):
                color = ''
                for i in range(6):
                    color_number = color_list[random.randint(0, 15)]
                    color += color_number
                color = '#' + color
                self.colors.append(color)

        mesh_airway = pv.read(
            os.path.join(save_dir,
                         case.split('.nii.gz')[0] + '.stl'))
        # pl = pv.Plotter(off_screen=True)
        # pl.add_mesh(mesh_airway, color='white', style='surface', opacity=0.4)
        skeleton_parse = cd = np.zeros(self.label.shape, dtype=np.int32)
        num = len(self.Bi)
        iii = 1
        for i in self.Bi:
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
        tree_parsing = tree_parsing_func(skeleton_parse, self.label, cd)
        end_time = time.time()


        pl = pv.Plotter(off_screen=True)
        pl.open_gif(os.path.join(save_dir, case.split('.nii.gz')[0] + '.gif'))
        points = []
        labels = []
        for k in range(1, num + 1):
            if np.sum(tree_parsing == k)==0:
                continue
            points.append(self.Bi_resize[
                k - 1]['member'][len(self.Bi_resize[k - 1]['member']) //
                                 2] if self.Bi_resize[k - 1]['member'] !=
                          [] else self.Bi_resize[k - 1]['end'])
            labels.append(self.Bi_resize[k - 1]['index'])
            iso = 0.95
            verts, faces, _, _ = marching_cubes(tree_parsing == k, iso)
            verts[:, 0] = verts[:, 0] - self.o[0]
            verts[:, 1] = verts[:, 1] - self.o[1]
            verts[:, 2] = verts[:, 2] - self.o[2]
            verts[:, 0] = verts[:, 0] * self.psize[0]
            verts[:, 1] = verts[:, 1] * self.psize[1]
            verts[:, 2] = verts[:, 2] * self.psize[2]
            faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
            mesh_seg = pv.PolyData(verts, faces)
            mesh_seg = mesh_seg.smooth(relaxation_factor=0.15)
            pl.add_mesh(mesh_seg, color=self.colors[k - 1], style='surface')  #

        n_frames = 60
        for i in range(n_frames):
            pl.camera_position = 'yz'
            pl.camera.azimuth = 360 * (i / n_frames)
            pl.render()
            pl.write_frame()
        pl.close()

        pl = pv.Plotter(off_screen=True)
        points = []
        labels = []
        for k in range(1, num + 1):
            if np.sum(tree_parsing == k)==0:
                continue
            points.append(self.Bi_resize[
                k - 1]['member'][len(self.Bi_resize[k - 1]['member']) //
                                 2] if self.Bi_resize[k - 1]['member'] !=
                          [] else self.Bi_resize[k - 1]['end'])
            labels.append(self.Bi_resize[k - 1]['index'])
            iso = 0.95
            verts, faces, _, _ = marching_cubes(tree_parsing == k, iso)
            verts[:, 0] = verts[:, 0] - self.o[0]
            verts[:, 1] = verts[:, 1] - self.o[1]
            verts[:, 2] = verts[:, 2] - self.o[2]
            verts[:, 0] = verts[:, 0] * self.psize[0]
            verts[:, 1] = verts[:, 1] * self.psize[1]
            verts[:, 2] = verts[:, 2] * self.psize[2]
            faces = np.c_[np.full(len(faces), 3), faces].astype(np.int32)
            mesh_seg = pv.PolyData(verts, faces)
            mesh_seg = mesh_seg.smooth(relaxation_factor=0.15)
            pl.add_mesh(mesh_seg, color=self.colors[k - 1], style='surface')  #


        pl.camera_position = 'yz'
        pl.show(screenshot=os.path.join(save_dir, case.split('.nii.gz')[0] + '_model.png'))

        return end_time

    def show_line1(self, save_dir, case):
        """
        Visualizes and saves the 3D airway mesh and segmented branches.

        Draws segmented branches with unique colors and adds them to the plot.
        Saves a screenshot of the visualization in the given directory.

        Parameters:
            save_dir (str): Directory to save the output screenshot.
            case (str): Case name used for the file paths.
        """
        if self.colors == []:
            color_list = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
                'C', 'D', 'E', 'F'
            ]
            for i in range(600):
                color = ''
                for i in range(6):
                    color_number = color_list[random.randint(0, 15)]
                    color += color_number
                color = '#' + color
                self.colors.append(color)

        mesh_airway = pv.read(
            os.path.join(save_dir,
                         case.split('.nii.gz')[0] + '.stl'))
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(mesh_airway, color='white', style='surface', opacity=0.4)
        for kk in range(len(self.Bi_resize)):
            i = self.Bi_resize[kk]
            bi = i['member'].copy()
            bi.insert(0, i['start'])
            if 'end' in i:
                bi.append(i['end'])
            bi = np.array(bi)
            if bi.shape[0] % 2 != 0:
                bi = bi[:-1, :]
            pl.add_lines(bi, color=self.colors[kk], width=5)
        pl.background_color = "white"
        pl.view_yz()
        screenshot_filename = os.path.join(save_dir,
                                           case.split('.nii.gz')[0] + '.png')
        pl.screenshot(screenshot_filename)
        pl.close()
