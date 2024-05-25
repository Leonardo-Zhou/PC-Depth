# PC_Depth

Official implementation of *Unsupervised Photometric-Consistent Depth Estimation from Endoscopic Monocular Video (PC-Depth)*


The video below shows the results of PC-Depth on the c3vd test sequence:

<img align="left" src="./readme.assets/demo.mp4" width="640"> 


## Dataset

### C3VD

> https://durrlab.github.io/C3VD/

Dataset Split for C3VD used in our experiment:

| model              | texture | video | frames  | stage  |
|--------------------|----|------|------|-----|
| Cecum              | 1  | a    | 276  | Train |
| Cecum              | 1  | b    | 765  | Train |
| Cecum              | 2  | b    | 1142 | Train |
| Cecum              | 2  | c    | 595  | Train |
| Cecum              | 3  | a    | 730  | Train |
| Cecum              | 4  | a    | 465  | Train |
| Cecum              | 4  | b    | 425  | Train |
| Sigmoid Colon      | 1  | a    | 700  | Train |
| Sigmoid Colon      | 2  | a    | 514  | Train |
| Sigmoid Colon      | 3  | b    | 536  | Train |
| Transcending Colon | 1  | a    | 61   | Train |
| Transcending Colon | 1  | b    | 700  | Train |
| Transcending Colon | 2  | b    | 103  | Train |
| Transcending Colon | 2  | c    | 235  | Train |
| Transcending Colon | 3  | a    | 250  | Train |
| Transcending Colon | 3  | b    | 214  | Train |
| Transcending Colon | 4  | a    | 382  | Train |
| Transcending Colon | 4  | b    | 597  | Train |
| Descending Colon   | 4  | a    | 148  | Verification |
| Cecum              | 2  | a    | 370  | Test |
| Sigmoid Colon      | 3  | a    | 613  | Test |
| Transcending Colon | 2  | a    | 194  | Test |


Our dataset structure is organized in alignment with the [SC-Depth](https://github.com/JiawangBian/sc_depth_pl) format. Since every sequence is captured using the same camera, we have placed the camera's intrinsic matrix within the train directory to ensure consistency and facilitate subsequent image processing and analysis.

Dataset structure used for training is:
```txt
/path/to/c3vd_data_root/
  train/
    cecum_t1_a/
    cecum_t1_b/
    desc_t4_a/
    sigmoid_t3_b/
      0.png   (RGB image)
      0.tiff  (gt depth)
      1.png
      1.tiff
      ...
    ...
    train.txt (containing training scene names)
    val.txt   (containing validation scene names)
    cam.txt   (3x3 camera intrinsic matrix)
  test/
    cecum_t2_a/
    sigmoid_t3_a/
    trans_t2_a/
```


### SCARED

> https://endovissub2019-scared.grand-challenge.org/

You can download the SCARED(endovis) dataset by signing the challenge rules and emailing them to max.allan@intusurg.com.

For the configuration of SCARED dataset, please refer to [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner)

We have updated the original code for the [SC-Depth](https://github.com/JiawangBian/sc_depth_pl) dataset to ensure its compatibility with receiving split-type inputs in the [monodepth2](https://github.com/nianticlabs/monodepth2) format. Additionally, the structure of the SCARED dataset remains consistent with [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner), and the specific split utilized for SCARED is located in the directory: `./splits/endovis`.

During training, you need to copy the files from `./splits/endovis` to the root directory of the SCARED dataset.

## Environment

Same as [SC-Depth](https://github.com/JiawangBian/sc_depth_pl).

## Train

We provide a bash script ("scripts/run_train_*.sh"), which shows how to train on C3VD, SCARED. 

You need to run these scripts from the root directory of the project to avoid any failures due to path issues.

## Test

1. Use `test/export_*_gt_depth.sh` to generate the ground truth of the corresponding dataset.
2. Use `test/export_*_pred_depth.sh` to obtain prediction results
3. Use `test/test_depth(pose)_npz.py` to test