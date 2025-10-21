import numpy as np
import torch
import os
import re
import cc3d
import matplotlib.pyplot as plt
import skimage.measure as measure
import SimpleITK as sitk
from matplotlib.pyplot import MultipleLocator
from torch.utils.data import DataLoader
from SE_UNet import SE_UNet
from data import SegTestCropData
from metrics import *
from util import *



def double_threshold_iteration(pred, h_thresh, l_thresh):
    neigb = np.array(
    [[-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0], [0, 1, 0], [1, -1, 0], [1, 0, 0], [1, 1, 0], [-1, -1, -1],
    [-1, 0, -1], [-1, 1, -1], [0, -1, -1], [0, 0, -1], [0, 1, -1], [1, -1, -1], [1, 0, -1], [1, 1, -1],
    [-1, -1, 1], [-1, 0, 1], [-1, 1, 1], [0, -1, 1], [0, 0, 1], [0, 1, 1], [1, -1, 1], [1, 0, 1],[1, 1, 1]])
    h, w,z = pred.shape
    pred = np.array(pred*255, dtype=np.float32)
    bin = np.where(pred >= h_thresh*255, 255, 0).astype(np.float32)
    gbin = bin.copy()
    gbin_pre = gbin-1
    while(gbin_pre.all() != gbin.all()):
        gbin_pre = gbin
        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if gbin[i][j][k] == 0 and pred[i][j][k] < h_thresh*255 and pred[i][j][k] >= l_thresh*255:
                        for n in range(0, 26):  # 26领域
                            inn = i + neigb[n, 0]
                            jnn = j + neigb[n, 1]
                            knn = k + neigb[n, 2]
                            if gbin[max(min(inn,h-1),0)][max(min(jnn,w-1),0)][max(min(knn,z-1),0)]:
                                gbin[i][j][k] = 255
                                break

    return gbin/255

def valid(log_path):
    with open(log_path, 'r') as file:
        lines=file.readlines()
    m_lines=[]
    for l in range(len(lines)):
        if l%3==1:
            m_lines.append(lines[l])
    TD=[]
    BD=[]
    DSC=[]
    Pre=[]
    SCORE=[]
    for l in m_lines:
        TD.append(float(re.findall(r'TD: (.*?) ', l)[0]))
        BD.append(float(re.findall(r'BD: (.*?) ', l)[0]))
        DSC.append(float(re.findall(r'DSC: (.*?) ', l)[0]))
        Pre.append(float(re.findall(r'Pre: (.*?) ', l)[0]))
    for l in range(len(TD)):
        SCORE.append((TD[l]+BD[l]+DSC[l]+Pre[l])/4)

    print('第',SCORE.index(max(SCORE)),'个epoch验证效果最佳，分数为',max(SCORE))
    return SCORE.index(max(SCORE))

def test(data_root, model_stage, best_epoch, testlog_savepath, result_savepath,
         DTI, file_path, file_root, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    model = SE_UNet(in_channel=2, n_classes=1)

    test_dataset = SegTestCropData(file_path,
                                   data_root,
                                   batch_size=8,
                                   cube_size=128,
                                   step=64)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=8,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True,
                                 drop_last=True)

    weights_dict = torch.load(model_stage +
                              '/SE_UNet_{}.pth'.format(best_epoch))
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    TDs, BDs, DSCs, Pres, Sens, Spes = [], [], [], [], [], []
    last_name = ''
    folder_name = model_stage.split('/')[-1]
    if not os.path.exists(result_savepath +'/' +folder_name):
        os.mkdir(result_savepath +'/' +folder_name)
    flag = False
    h_thresh = 0.5
    l_thresh = 0.35
    with torch.no_grad():
        for i, (x, name, pos) in enumerate(test_dataloader):  #8个case18
            name = name[0]
            if name != last_name:
                if last_name != '':
                    print(last_name)
                    pred = pred / pred_num
                    pred = pred[0, 0]
                    if DTI == 0:
                        pred[pred >= 0.5] = 1
                        pred[pred < 0.5] = 0
                    else:
                        pred = double_threshold_iteration(
                            pred, h_thresh, l_thresh)
                        
                    pred[0:int(0.15 * pred.shape[0]), :, :] = 0
                    pred[int(0.85 * pred.shape[0]):, :, :] = 0
                    pred[:, 0:int(0.15 * pred.shape[1]), :] = 0
                    pred[:, int(0.85 * pred.shape[1]):, :] = 0

                    pred = maximum_3d(pred)
                    pred_img = sitk.GetImageFromArray(pred.astype(np.byte))
                    pred_img.SetOrigin(img.GetOrigin())
                    pred_img.SetDirection(img.GetDirection())
                    pred_img.SetSpacing(img.GetSpacing())
                    sitk.WriteImage(
                        pred_img,
                        os.path.join(result_savepath, folder_name,
                                     last_name + '.nii.gz'))

                    TD, BD, DSC, Pre, Sen, Spe = evaluation_case_test(
                        pred, label, last_name, testlog_savepath, file_root)
                    TDs.append(TD)
                    BDs.append(BD)
                    DSCs.append(DSC)
                    Pres.append(Pre)
                    Sens.append(Sen)
                    Spes.append(Spe)
                img = sitk.ReadImage(data_root + '/data/' + name + 'data_cut' +
                                     '.nii.gz')
                arr = sitk.GetArrayFromImage(img)
                label = sitk.ReadImage(data_root + '/mask/' + name +
                                       'mask_cut' + '.nii.gz')
                label = sitk.GetArrayFromImage(label)
                pred = np.zeros(arr.shape)
                pred = pred[np.newaxis, np.newaxis, ...]
                pred_num = np.zeros(pred.shape)
                last_name = name
            #
            x = x.cuda()
            p0, p = model(x)
            p = torch.sigmoid(p)
            p = p.cpu().detach().numpy()
            pos = pos.numpy()
            for i in range(len(pos)):
                # print(pos)
                xl, xr, yl, yr, zl, zr = pos[i, 0], pos[i, 1], pos[i, 2], pos[
                    i, 3], pos[i, 4], pos[i, 5]
                pred[0, :, xl:xr, yl:yr, zl:zr] += p[i]
                pred_num[0, :, xl:xr, yl:yr, zl:zr] += 1
        print(last_name)
        pred = pred / pred_num
        pred = pred[0, 0]
        if DTI == 0:
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
        else:
            pred = double_threshold_iteration(pred, h_thresh, l_thresh)
        
        pred[0:int(0.15 * pred.shape[0]), :, :] = 0
        pred[int(0.85 * pred.shape[0]):, :, :] = 0
        pred[:, 0:int(0.15 * pred.shape[1]), :] = 0
        pred[:, int(0.85 * pred.shape[1]):, :] = 0

        pred = maximum_3d(pred)
        pred_img = sitk.GetImageFromArray(pred.astype(np.byte))
        pred_img.SetOrigin(img.GetOrigin())
        pred_img.SetDirection(img.GetDirection())
        pred_img.SetSpacing(img.GetSpacing())
        sitk.WriteImage(
            pred_img,
            os.path.join(result_savepath, folder_name, last_name + '.nii.gz'))

        TD, BD, DSC, Pre, Sen, Spe = evaluation_case_test(
            pred, label, last_name, testlog_savepath, file_root)
        TDs.append(TD)
        BDs.append(BD)
        DSCs.append(DSC)
        Pres.append(Pre)
        Sens.append(Sen)
        Spes.append(Spe)

        TD_mean = np.mean(TDs)
        TD_std = np.std(TDs)
        BD_mean = np.mean(BDs)
        BD_std = np.std(BDs)
        DSC_mean = np.mean(DSCs)
        DSC_std = np.std(DSCs)
        Pre_mean = np.mean(Pres)
        Pre_std = np.std(Pres)
        Sen_mean = np.mean(Sens)
        Sen_std = np.std(Sens)
        Spe_mean = np.mean(Spes)
        Spe_std = np.std(Spes)
        print(
            "TD: %0.4f (%0.4f), BD: %0.4f (%0.4f), DSC: %0.4f (%0.4f), Pre: %0.4f (%0.4f), Sen: %0.4f (%0.4f), Spe: %0.4f (%0.4f)"
            % (TD_mean, TD_std, BD_mean, BD_std, DSC_mean, DSC_std, Pre_mean,
               Pre_std, Sen_mean, Sen_std, Spe_mean, Spe_std))
        line = "TD: %0.4f (%0.4f), BD: %0.4f (%0.4f), DSC: %0.4f (%0.4f), Pre: %0.4f (%0.4f), Sen: %0.4f (%0.4f), Spe: %0.4f (%0.4f)" % (
            TD_mean, TD_std, BD_mean, BD_std, DSC_mean, DSC_std, Pre_mean,
            Pre_std, Sen_mean, Sen_std, Spe_mean, Spe_std)
        with open(testlog_savepath, 'a') as file:
            file.writelines([line + '\n'])
        #画图
        metrics = [TDs, BDs, DSCs, Pres, Sens, Spes]
        plt.figure(figsize=(10, 10))
        b = plt.boxplot(metrics,
                        meanline=True,
                        showmeans=True,
                        labels=['TD', 'BD', 'DSC', 'Pre', 'Sen', 'Spe'],
                        patch_artist=True)
        for c in b["boxes"]:
            c.set(color=random_color())
        plt.grid(linestyle='-.')
        plt.title('Metrics of ' + folder_name, fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        y_major_locator = MultipleLocator(10)
        #把y轴的刻度间隔设置为10，并存在变量里
        ax = plt.gca()
        #ax为两条坐标轴的实例
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(0, 105)
        #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
        plt.savefig('Metrics of ' + folder_name + '.png')
        plt.show()
        # np.save('metrics of '+whichstage+'.npy', metrics)

def evaluation_case_test(pred, label, name,testlog_savepath,file_root):
    parsing = sitk.ReadImage(os.path.join(file_root, 'tree_parse_test', name + 'mask_cut.nii.gz'))
    parsing = sitk.GetArrayFromImage(parsing)
    # parsing=parsing.transpose(2,1,0)
    if len(pred.shape) > 3:
        pred = pred[0]
    if len(label.shape) > 3:
        label = label[0]

    cd=cc3d.connected_components(pred, connectivity=26)
    region = measure.regionprops(cd)
    num_list = [i for i in range(1, np.max(cd)+1)]
    area_list = [region[i-1].area for i in num_list]
    volume_sort = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    if volume_sort!=[]:
        large_cd =(cd==volume_sort[0]).astype(np.uint8)
    else:
        large_cd=pred.astype(np.uint8)

    skeleton = sitk.ReadImage(os.path.join(file_root, 'skeleton_test', name + 'mask_cut.nii.gz'))
    skeleton = sitk.GetArrayFromImage(skeleton)
    # skeleton=skeleton.transpose(2,1,0)
    skeleton = (skeleton > 0)
    skeleton = skeleton.astype('uint8')

    _,_,BD=branch_detected_calculation(large_cd, parsing, skeleton)
    DSC=dice_coefficient_score_calculation(large_cd, label)
    TD=tree_length_calculation(large_cd, skeleton)
    Sen=sensitivity_calculation(large_cd, label)
    Spe=specificity_calculation(large_cd, label)
    Pre=precision_calculation(large_cd, label)

    print(name, "TD: %0.4f" % (TD),"BD: %0.4f" % (BD), "DSC: %0.4f" % (DSC),"Precision: %0.4f" % (Pre), "Sen: %0.4f" % (Sen),
          "Spe: %0.4f" % (Spe))
    return TD,BD,DSC,Pre,Sen,Spe

if __name__ == '__main__':
    ###################################### GPU setup
    gpu_all_num = 8
    gpu_need_num = 4
    lm_per_gpu = 20000
    while 1:
        free = np.zeros((gpu_all_num))
        for i in range(gpu_all_num):
            free[i] = get_gpu_mem_info(i)
        if len(np.where(free > lm_per_gpu)[0]) >= gpu_need_num:
            break
    gpu = ','.join(
        [str(x) for x in list(np.where(free > lm_per_gpu)[0][0:gpu_need_num])])

    log_path = './LOG/log_stage_three.txt'
    best_epoch = valid(log_path)
    model_stage = './saved_model/stage_three'

    data_root = '/mnt/yby/AFTER_PREPROCESS'
    file_path = './data/test.json'
    file_root = './data'
    result_savepath = './test_result'
    testlog_savepath = './LOG/testlog_stage_three.txt'
    DTI = 1

    test(data_root,model_stage, best_epoch, testlog_savepath, result_savepath, DTI,
         file_path, file_root,gpu)
