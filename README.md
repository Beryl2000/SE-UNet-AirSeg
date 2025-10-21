#  Scale-Enhanced U-Net (SE-UNet)

This repository provides the official implementation for the paper:  
**"Progressive Curriculum Learning with Scale-Enhanced U-Net for Continuous Airway Segmentation"**  
by *Bingyu Yang, Qingyao Tian, Huai Liao, Xinyan Huang, Jinlin Wu, Jingdi Hu, and Hongbin Liu*.

> This work proposes a **Progressive Curriculum Learning** framework combined with a **Scale-Enhanced U-Net (SE-UNet)** to achieve **continuous and topology-preserving airway segmentation** from chest CT scans.  
> The model addresses the intra-class imbalance between large and small bronchial branches and the difficulty of segmenting blurred airway structures.  
> Our method introduces a **three-stage curriculum strategy** and **scale-enhanced feature aggregation**, significantly improving small-airway detection and overall airway continuity.  
---

## ⚙️ Installation

### Create environment
```bash
conda env create -f environment.yml
conda activate seunet
```

### Or install manually
```bash
pip install -r requirements.txt
```

Required packages include: `torch`, `numpy`, `scipy`, `SimpleITK`, `scikit-image`, `pyvista`, and `cc3d`.

---

## 📘 Overview

- **Progressive Curriculum Learning:**  
  Three-stage training from coarse to fine airway segmentation.

- **Scale-Enhanced U-Net (SE-UNet):**  
  Multi-scale encoder–decoder with feature enhancement and detail injection modules.

- **Airway Tree Parsing:**  
  Post-processing stage reconstructing 3D topology for branch-level analysis.

---

## 🧩 Main Pipeline

### Step-by-step Usage

1. **Preprocess the dataset**
   ```bash
   python preprocessing.py
   ```
   Perform CT data normalization, cropping, and initial cleaning.

2. **Generate dataset split and config**
   ```bash
   python write_json.py
   ```
   Create training/validation/test split and JSON configuration files.

3. **Compute prior weights**
   ```bash
   python lib_weight.py
   ```
   Calculate local-size-based airway prior weights. Results are stored in `./data/LIB_weight`.

4. **Skeleton and parsing**
   ```bash
   python ske_and_parse.py
   ```
   Extract skeletons and perform branch parsing. Results are saved to `./data/tree_parse(_val/_test)` and `./data/skeleton(_val/_test)`.

5. **Progressive three-stage training**
   ```bash
   python train.py
   ```
   - Stage 1: Train on coarse airway structures → outputs `./data/pred_1`  
   - Stage 2: Train with refined small-airway weighting → outputs `./data/pred_2`, `./data/BR_weight`, `./data/br_skel`  
   - Stage 3: Final continuity refinement training.

6. **Testing**
   ```bash
   python test.py
   ```
   Evaluate the final model on the test set using the best checkpoint.

7. **Prediction / Deployment**
   ```bash
   python prediction.py
   ```
   Deploy trained models for clinical inference.

---

## 🌲 Airway Tree Parsing

This module analyzes segmented airway structures to obtain topology metrics such as total length, number of branches, and connectivity.

Two parsing algorithms are included: **Ours** and **ATM’22** ([ATM22 repo](https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work/tree/main/evaluation)).

### Run Ours
```bash
python tree_parsing.py --pred_mask_path ./demo_mask/ --save_path ./demo_output_Ours/ --merge_t 5
```

### Run ATM22
```bash
python tree_parsing.py --pred_mask_path ./demo_mask/ --save_ATM22_path ./demo_output_ATM22/
```

---


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

## 🧠 Citation

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

## 🔮 Future Updates

- [x] ~~Sharing data splits and preprocessing scripts.~~
- [x] ~~Training code~~

For any questions, feel free to open an issue or contact us.


<!-- 

git config --global http.proxy  http://127.0.0.1:16159
git config --global https.proxy https://127.0.0.1:16159

git add .
git commit -m "updata"
git push origin master --force 

-->