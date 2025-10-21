import os
import numpy as np
import torch
import pyvista as pv
from skimage.morphology import skeletonize_3d
from skimage.measure import marching_cubes_lewiner
from stl import mesh
from SE_UNet import SE_UNet
from preprocessing import preprocess_CT,load_itk_image,save_itk
from util import *
import time

def double_threshold_iteration(pred, h_thresh, l_thresh):
    neigb = np.array(
    [[-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0], [0, 1, 0], [1, -1, 0], [1, 0, 0], [1, 1, 0], [-1, -1, -1],
    [-1, 0, -1], [-1, 1, -1], [0, -1, -1], [0, 0, -1], [0, 1, -1], [1, -1, -1], [1, 0, -1], [1, 1, -1],
    [-1, -1, 1], [-1, 0, 1], [-1, 1, 1], [0, -1, 1], [0, 0, 1], [0, 1, 1], [1, -1, 1], [1, 0, 1],[1, 1, 1]])
    h, w,z = pred.shape
    pred = np.array(pred*255, dtype=np.float64)
    bin = np.where(pred >= h_thresh*255, 255, 0).astype(np.float64)
    gbin = bin.copy()
    gbin_pre = gbin-1
    while(gbin_pre.all() != gbin.all()):
        gbin_pre = gbin
        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if gbin[i][j][k] == 0 and pred[i][j][k] < h_thresh*255 and pred[i][j][k] >= l_thresh*255:
                        for n in range(0, 26):  
                            inn = i + neigb[n, 0]
                            jnn = j + neigb[n, 1]
                            knn = k + neigb[n, 2]
                            if gbin[max(min(inn,h-1),0)][max(min(jnn,w-1),0)][max(min(knn,z-1),0)]:
                                gbin[i][j][k] = 255
                                break

    return gbin/255

def two_channel(data):
    data = data.astype(float)
    data2 = data.copy()
    data2[data2 > 500] = 500
    data2[data2 < -1000] = -1000
    data2 = (data2 + 1000) / 1500
    data[data > 1024] = 1024
    data[data < -1024] = -1024
    data = (data + 1024) / 2048

    return data, data2

def network_prediction(model_path,patient,data_path, save_path,flip=False,gpu='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    start_time = time.time()
    print ('starting segmenting lung CT')

    model = SE_UNet(in_channel=2, n_classes=1)
    weights_dict = torch.load(model_path)
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    cube_size = 128
    step = 64

    img, origin, spacing = load_itk_image(data_path+patient.split('data')[0]+'data_cut.nii.gz')
    img=img-1024
    
    # imput ct img
    img=np.expand_dims(img,axis=0)
    img, img2 = two_channel(img)
    img = np.array([img, img2])
    x=torch.from_numpy(img.astype(np.float32))
    x = x.transpose(0, 1)
    x = x.cuda()
    pred = np.zeros((1,1,x.shape[2],x.shape[3],x.shape[4]))
    pred_num = np.zeros((1,1,x.shape[2],x.shape[3],x.shape[4]))
    xnum = (x.shape[2]-cube_size)//step + 1 if (x.shape[2]-cube_size)%step==0 else (x.shape[2]-cube_size)//step + 2
    ynum = (x.shape[3]-cube_size)//step + 1 if (x.shape[3]-cube_size)%step==0 else (x.shape[3]-cube_size)//step + 2
    znum = (x.shape[4]-cube_size)//step + 1 if (x.shape[4]-cube_size)%step==0 else (x.shape[4]-cube_size)//step + 2
    for xx in range(xnum):
        xl = step*xx
        xr = step*xx + cube_size
        if xr > x.shape[2]:
            xr = x.shape[2]
            xl = x.shape[2]-cube_size
        for yy in range(ynum):
            yl = step*yy
            yr = step*yy + cube_size
            if yr > x.shape[3]:
                yr = x.shape[3]
                yl = x.shape[3] - cube_size
            for zz in range(znum):
                zl = step*zz
                zr = step*zz + cube_size
                if zr > x.shape[4]:
                    zr = x.shape[4]
                    zl = x.shape[4] - cube_size

                x_input = x[:,:,xl:xr,yl:yr,zl:zr]
                p0, p = model(x_input)
                p = torch.sigmoid(p)
                p = p.cpu().detach().numpy()
                pred[:,:,xl:xr,yl:yr,zl:zr] += p
                pred_num[:,:,xl:xr,yl:yr,zl:zr] += 1

    pred = pred/pred_num
    pred = np.squeeze(pred)
    pred=double_threshold_iteration(pred,h_thresh=0.5,l_thresh=0.4)
    pred[0:int(0.15 * pred.shape[0]), :, :] = 0
    pred[int(0.85 * pred.shape[0]):, :, :] = 0
    pred[:, 0:int(0.15 * pred.shape[1]), :] = 0
    pred[:, int(0.85 * pred.shape[1]):, :] = 0
    pred_img=maximum_3d(pred)

    result=pred_img
    save_itk(result.astype(np.byte),origin,spacing, save_path+patient+'_pred_mask.nii.gz')

    if flip==True:
        for y in range(result.shape[1]):
            result[ :,y, :] = np.flipud(result[ :,y, :])

    iso = 0.95
    verts, faces, _, _ = marching_cubes_lewiner(result, iso)
    skel = skeletonize_3d(result)
    B = np.array(np.where(skel != 0))
    B = B[:, B[2].argsort()]
    B = B.T
    B=B.astype(np.float32)
    ox=np.mean(B[:,0])
    oy=np.mean(B[:,1])
    oz=np.mean(B[:,2])
    verts[:, 0] = verts[:, 0]-ox
    verts[:, 1] = verts[:, 1]-oy
    verts[:, 2] = verts[:, 2]-oz
    verts[:, 0] = verts[:, 0] * spacing[0]/10
    verts[:, 1] = verts[:, 1] * spacing[1]/10
    verts[:, 2] = verts[:, 2] * spacing[2]/10
    cube_a = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)) 
    for i, f in enumerate(faces):
        for j in range(3):
            cube_a.vectors[i][j] = verts[f[j], :]
    cube_a.save(save_path + "/" +patient+'.stl')
    mesh_a = pv.read(save_path + "/" +patient+'.stl') 
    mesh_a_smooth = mesh_a.smooth(relaxation_factor=0.2)
    mesh_a_smooth.plot(background='w')
    mesh_a_smooth.save(save_path + "/" +patient+'.stl')

    end_time = time.time()
    print ('end segmenting lung CT, time %d seconds'%(end_time-start_time))

    return result,spacing

if __name__ == '__main__':
    gpu_all = 1
    gpu_need = 1
    lm_per_gpu = 20000
    while 1:
        free = np.zeros((gpu_all))
        for i in range(gpu_all):
            free[i] = get_gpu_mem_info(i)
        if len(np.where(free > lm_per_gpu)[0]) >= gpu_need:
            break
    gpu = ','.join(
        [str(x) for x in list(np.where(free > lm_per_gpu)[0][0:gpu_need])])



    model_path = 'saved_model/stage_three/SE_UNet_43.pth'
    DCM_Path = 'example_dcm'
    flist = os.listdir(DCM_Path)
    flist.sort()
    for patient in flist:
        print('ct：', patient)
        data_path = os.path.join(DCM_Path , patient)
        save_path = './processed_cases/'
        preprocess_CT(data_path,save_path,format='nii.gz',mode='prediction',multides=False)

        data_path = './processed_cases/'
        save_path = './predicted_airways/'
        pred, spacing = network_prediction(model_path,
                                            patient,
                                            data_path,
                                            save_path,
                                            flip=True)


    sfd = 9
