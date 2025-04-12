#!/bin/bash

cd /home/work/unknownbzop1/ai-programming

for model in cnn cnn2 fcnn
do
    for batch in 8 16 32 64
    do
        for lr in 0.00003 0.0001 0.0003 0.001 0.003
        do
            python3 cifar_training.py --model_type $model --batch_size $batch --epochs 20 --lr $lr
        done
        git add .
        git commit -m date +"%y%m%d-%H%M updates"
        git push origin main
    done
done