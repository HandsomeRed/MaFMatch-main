#!/bin/bash

#now=$(date +"%Y%m%d_%H%M%S")

# dataset: ['acdc','la']
# method: ['MaFmatch', 'fixmatch']
# exp: just for specifying the 'save_path'
# split: ['1', '3', '7'] or ['20%','50%']
dataset='acdc'
method='MaFmatch'
exp='unet'
split='3'

now=$(date +%Y%m%d_%H%M%S)

config=configs/$dataset.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split/$now

mkdir -p $save_path

python -m torch.distributed.run \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port 29500 | tee $save_path/$now.log
