#!/bin/bash

cd /home/work/unknownbzop1/ai-programming || return
cp .gitconfig ../..

for wd in 1e-5 3e-5 1e-4 3e-4 1e-3
do
    for batch in 256 128 64 32
    do
        for lr in 0.1 0.03 0.01 0.003 0.001
        do
            python3 train_cifar100_resnet.py --weight_decay $wd --batch_size $batch --lr $lr --patience 7
        done
        git add .
        git commit -m date +"%y%m%d-%H%M updates"
        git push origin main
    done
done