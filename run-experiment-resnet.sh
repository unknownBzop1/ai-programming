#!/bin/bash

cd /home/work/unknownbzop1/ai-programming || return
cp .gitconfig ../..

for sch in plateau
do
    for batch in 16 32 64
    do
        for lr in 0.003 0.01 0.03 0.1
        do
            python3 train_cifar100_resnet.py --weight_decay 1e-4 --batch_size $batch --lr $lr --patience 7 --scheduler $sch
        done
        git add .
        git commit -m date +"%y%m%d-%H%M updates"
        git push origin main
    done
done

for sch in step
do
    for batch in 16 32 64
    do
        for lr in 0.03 0.1 0.3
        do
            python3 train_cifar100_resnet.py --weight_decay 1e-4 --batch_size $batch --lr $lr --patience 7 --scheduler $sch
        done
        git add .
        git commit -m date +"%y%m%d-%H%M updates"
        git push origin main
    done
done