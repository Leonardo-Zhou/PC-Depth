# PC_Depth

> Official implementation of PC-Depth

## Dataset

### C3VD
> https://durrlab.github.io/C3VD/

Dataset Split for C3VD used in our experiment.

| model              | texture | video | frames  | stage  |
|--------------------|----|------|------|-----|
| Cecum              | 1  | a    | 276  | Train |
| Cecum              | 1  | a    | 276  | Train |
| Cecum              | 1  | b    | 765  | Train |
| Cecum              | 2  | b    | 1120 | Train |
| Cecum              | 2  | c    | 595  | Train |
| Cecum              | 3  | a    | 730  | Train |
| Cecum              | 4  | a    | 465  | Train |
| Cecum              | 4  | b    | 425  | Train |
| Sigmoid Colon      | 1  | a    | 800  | Train |
| Sigmoid Colon      | 2  | a    | 513  | Train |
| Sigmoid Colon      | 3  | b    | 536  | Train |
| Transcending Colon | 1  | a    | 61   | Train |
| Transcending Colon | 1  | b    | 700  | Train |
| Transcending Colon | 2  | b    | 102  | Train |
| Transcending Colon | 2  | c    | 235  | Train |
| Transcending Colon | 3  | a    | 250  | Train |
| Transcending Colon | 3  | b    | 214  | Train |
| Transcending Colon | 4  | a    | 384  | Train |
| Transcending Colon | 4  | b    | 595  | Train |
| Descending Colon   | 4  | a    | 148  | Verification |
| Cecum              | 2  | a    | 370  | Test |
| Sigmoid Colon      | 3  | a    | 610  | Test |
| Transcending Colon | 2  | a    | 194  | Test |

Dataset structure used for training:


### SCARED
> https://endovissub2019-scared.grand-challenge.org/

You can download the SCARED(endovis) dataset by signing the challenge rules and emailing them to max.allan@intusurg.com.

For the configuration of SCARED data set, please refer to [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner)

SCARED use split for training: ./splits/endovis

## Environment

Same as [SC-Depth](https://github.com/JiawangBian/sc_depth_pl).

## Train

We provide a bash script ("scripts/run_train.sh"), which shows how to train on C3VD, SCARED.

## Test

1. Use test/export_*_gt_depth to generate the ground truth of the corresponding dataset.
2. Use test/export_*_pred_depth to obtain prediction results
3. Use test/test_depth(pose)_npz.py to test