# Airway Tree Parsing

This repository contains the code for airway tree parsing and the comparison results of the algorithms. Two algorithms are used: **Ours** and **ATM22**(https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work/tree/main/evaluation). The goal of the project is to parse the segmented airway tree structures from lung CT scans into meaningful representations. This allows for the evaluation of tree length and branching numbers of segmentation algorithms and provides support for clinical analysis.

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
python tree_parsing.py --pred_mask_path ./demo_mask/ --save_path ./demo_output_Ours/ --merge_t 5
```
This command will load predicted mask files from `./demo_mask/` and save the processed results to `./demo_output_Ours/`.`--merge_t`: Threshold for merging branches during airway skeleton parsing (default: 5).


2. Run atm22_skel_parse Only

To run only atm22_skel_parse and save the output, use the following command:

```bash
python tree_parsing.py --pred_mask_path ./demo_mask/ --save_ATM22_path ./demo_output_ATM22/

```
This command will load predicted mask files from `./demo_mask/` and save the processed results to `./demo_output_ATM22/`.

<!-- ## Results

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


It is evident that the airway tree parsing method from the ATM22 challenge is not only inefficient but also generates inaccurate airway segmentations. In contrast, our method delivers results that are both efficient and reliable. -->

## Citation

If you use this project in your research, please cite the following papers:

```bibtex
@article{yang2024multi,
  title={Multi-Stage Airway Segmentation in Lung CT Based on Multi-scale Nested Residual UNet},
  author={Yang, Bingyu and Liao, Huai and Huang, Xinyan and Tian, Qingyao and Wu, Jinlin and Hu, Jingdi and Liu, Hongbin},
  journal={arXiv preprint arXiv:2410.18456},
  year={2024}
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


<!-- git config --global http.proxy  http://127.0.0.1:16541
git config --global https.proxy https://127.0.0.1:16541

git add .
git commit -m "updata"
git push origin master --force -->