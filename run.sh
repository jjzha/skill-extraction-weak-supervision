#! /bin/bash

SET=dev

for model in JobBERT RoBERTa
do
    for embed in ISO AOC WSE
    do
        python weak_supervision.py --site sayfullina --set $SET --model $model --embed_type $embed --contextual
        python weak_supervision.py --site sayfullina --set $SET --model $model --embed_type $embed
        for thresh in `seq 0 0.1 1`
        do
            for site in tech house
            do
                python weak_supervision.py --site $site --set $SET --model $model --embed_type $embed --threshold $thresh --contextual
                python weak_supervision.py --site $site --set $SET --model $model --embed_type $embed --threshold $thresh
            done
        done
    done
done