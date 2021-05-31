#!/usr/bin/env bash
python train.py --seed 0 --GCN --X --XF --GXF --train-size 5  --weight-decay 5e-6 --repeat 50 --dataset 20news
python train.py --seed 0 --GCN --X --XF --GXF --train-size 5  --weight-decay 5e-5 --repeat 50 --dataset large_cora
python train.py --seed 0 --GCN --X --XF --GXF --train-size 5  --weight-decay 5e-7 --repeat 50 --dataset wiki

python train.py --seed 0 --GCN --X --XF --GXF --train-size 20 --weight-decay 5e-6 --repeat 50 --dataset 20news
python train.py --seed 0 --GCN --X --XF --GXF --train-size 20 --weight-decay 5e-5 --repeat 50 --dataset large_cora
python train.py --seed 0 --GCN --X --XF --GXF --train-size 20 --weight-decay 5e-7 --repeat 50 --dataset wiki

python train.py --seed 0 --GCN --XF --GXF --train-size 0.05 --valid-size 0.3 --repeat 50 --dataset webkb_cornell
python train.py --seed 0 --GCN --XF --GXF --train-size 0.05 --valid-size 0.3 --repeat 50 --dataset webkb_texas
python train.py --seed 0 --GCN --XF --GXF --train-size 0.05 --valid-size 0.3 --repeat 50 --dataset webkb_wisconsin
python train.py --seed 0 --GCN --XF --GXF --train-size 0.05 --valid-size 0.3 --repeat 50 --dataset webkb_washington
