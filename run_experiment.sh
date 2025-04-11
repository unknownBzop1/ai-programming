#!/bin/bash

for model in cnn fcnn resnet18 googlenet
do
    for batch in 16 32 64 128
    do
        for epoch in 5 10 20 40
        do
            for lr in 0.0001 0.001 0.01 0.1
            do
                python3 cifar_training.py --model_type $model --batch_size $batch --epochs $epoch --lr $lr
            done
        done
    done
done