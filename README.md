# Progressive Curriculum Learning with Scale-Enhanced U-Net for Continuous Airway Segmentation


This repository provides the implementation of our paper:  
**Progressive Curriculum Learning with Scale-Enhanced U-Net for Continuous Airway Segmentation**  
[arXiv link](https://arxiv.org/abs/2410.18456)

Continuous and accurate segmentation of airways in chest CT images is crucial for preoperative planning and real-time bronchoscopy navigation. This project contains the codebase for our progressive curriculum learning pipeline, the Scale-Enhanced U-Net (SE-UNet), and airway tree parsing and evaluation.

> ⚠️ **Note**: Currently, this repository includes the released code for **airway tree parsing**.  
> The **training, validation, and testing code** for the full airway segmentation pipeline (SE-UNet with progressive curriculum learning and ATRL loss) will be uploaded and continuously updated.

---

## Airway Tree Parsing

The airway tree parsing module provides code for extracting centerlines, parsing airway structures, and evaluating tree length/branching. We compare our method against the widely used **ATM22** approach.  

Two algorithms are provided:
- **Ours**
- **ATM22** (official code: [ATM-22-Related-Work](https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work/tree/main/evaluation))

Parsing the airway tree structures from lung CT masks enables evaluation of branch connectivity, tree length, and other clinically meaningful indices.


## Requirements

- Python 3.11.10
- `numpy`
- `scipy`
- `pyvista`
- `skimage`
- `SimpleITK`
- `cc3d`

You can install the necessary packages via pip:

```bash
pip install -r requirements.txt
```

## Code

First, place the mask of the airway tree that needs to be parsed in the folder `./demo_mask/` in the `.nii.gz` format.

1. Run ours_skel_parse Only

To run only ours_skel_parse and save the output, use the following command:

```bash
python Main.py --pred_mask_path ./demo_mask/ --save_path ./demo_output_Ours/ --merge_t 5
```
This command will load predicted mask files from `./demo_mask/` and save the processed results to `./demo_output_Ours/`.`--merge_t`: Threshold for merging branches during airway skeleton parsing (default: 5).


2. Run atm22_skel_parse Only

To run only atm22_skel_parse and save the output, use the following command:

```bash
python Main.py --pred_mask_path ./demo_mask/ --save_ATM22_path ./demo_output_ATM22/

```
This command will load predicted mask files from `./demo_mask/` and save the processed results to `./demo_output_ATM22/`.

## Results

The following table provides a detailed comparison of our method with the widely-used ATM22 challenge method on the demo data:

| Method             | Case      | Centerline segment time | Airway tree parse time | Num of branches |
|--------------------|--------------|-----------------------|-------------|---------------|
| Ours    | CASE073       | 12s                | 14s   | 274        |
| ATM22    | CASE073       | 38s                | 322s   | 298         |


<div style="text-align: center;">
  <img src="./demo_output_Ours/CASE073_model.png" alt="CASE073-Ours" width="45%">
  <div><b>CASE073 - Ours</b></div>
</div>

<div style="text-align: center;">
  <img src="./demo_output_Ours/CASE073.gif" alt="CASE073-Ours" width="45%">
  <div><b>CASE073 - Ours</b></div>
</div>


<div style="text-align: center;">
  <img src="./demo_output_ATM22/CASE073_model.png" alt="CASE073-ATM22" width="45%">
  <div><b>CASE073 - ATM22</b></div>
</div>

<div style="text-align: center;">
  <img src="./demo_output_ATM22/CASE073.gif" alt="CASE073-ATM22" width="45%">
  <div><b>CASE073 - ATM22</b></div>
</div>


It is evident that the airway tree parsing method from the ATM22 challenge is not only inefficient but also generates inaccurate airway segmentations. In contrast, our method delivers results that are both efficient and reliable.

## Citation

If you use this project in your research, please cite the following papers:

```bibtex
@misc{yang2025progressivecurriculumlearningscaleenhanced,
      title={Progressive Curriculum Learning with Scale-Enhanced U-Net for Continuous Airway Segmentation}, 
      author={Bingyu Yang and Qingyao Tian and Huai Liao and Xinyan Huang and Jinlin Wu and Jingdi Hu and Hongbin Liu},
      year={2025},
      eprint={2410.18456},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2410.18456}, 
}
@article{zhang2023multi,
  title={Multi-site, multi-domain airway tree modeling},
  author={Zhang, Minghui and Wu, Yangqian and Zhang, Hanxiao and Qin, Yulei and Zheng, Hao and Tang, Wen and Arnold, Corey and Pei, Chenhao and Yu, Pengxin and Nan, Yang and others},
  journal={Medical image analysis},
  volume={90},
  pages={102957},
  year={2023},
  publisher={Elsevier}
}
```
