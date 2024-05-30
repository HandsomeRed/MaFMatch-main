# MaFMatch

We provide the official PyTorch implementation of our MaFMatch:

> Make Weak-to-Strong Consistency Work Better for Semi-supervised Medical Image Segmentation*.*
>
> _by Jianwu Long, Yuwei Li_\*

### Requirements

This repository utilizes PyTorch 1.12.1, CUDA 11.3, and Python 3.10. All experiments presented in our paper were performed on a single NVIDIA 3080 GPU under consistent experimental conditions.

### Dataset

- ACDC: [image and mask](https://drive.google.com/file/d/1LOust-JfKTDsFnvaidFOcAVhkZrgW1Ac/view?usp=sharing)

Please modify your dataset path in configuration files.

    ├── [Your ACDC Path]
        └── data

### Settings

Modify the `method` from `'MaFmatch'` to `'fixmatch'` in ’train.sh‘.

## How to start

### Usage

1.  Clone the repository.;

<!---->

    git clone https://github.com/HandsomeRed/MaFMatch-main

2.  Train the model;

<!---->

    cd MaFMatch-main

    sh train.sh

3.  Test the model;

    We have uploaded the MaFMatch weight files for testing on 5% ACDC annotated data. (Note: The values are slightly higher than those in the paper as the paper reports the average values.)

<!---->

    cd MaFMatch-main

    sh test.sh

### Acknowledgements:

Our code is adapted from [unimatch](https://github.com/LiheYoung/UniMatch), and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). We appreciate the valuable contributions of these authors and hope our model can further advance related research.

### Questions

If any questions, feel free to contact me at 'tinyred.li\@foxmail.com'
