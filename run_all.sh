#!/bin/bash

echo "======================================"
echo "SO-fore2-1 (task_mode=0) - Training"
echo "======================================"
CKPT0=$(python train_dl.py --task_mode 0 | grep "ckpt/" | tail -1)
echo "[SO-fore2-1] Best checkpoint = $CKPT0"

echo "======================================"
echo "SO-fore2-1 (task_mode=0) - Forecast"
echo "======================================"
python forecast_dl.py --task_mode 0 --ckpt_path $CKPT0



echo "======================================"
echo "SO-fore2-10 (task_mode=1) - Training"
echo "======================================"
CKPT1=$(python train_dl.py --task_mode 1 | grep "ckpt/" | tail -1)
echo "[SO-fore2-10] Best checkpoint = $CKPT1"

echo "======================================"
echo "SO-fore2-10 (task_mode=1) - Forecast"
echo "======================================"
python forecast_dl.py --task_mode 1 --ckpt_path $CKPT1



echo "======================================"
echo "Demand SO-fore (new_product=1) - Training"
echo "======================================"
CKPT2=$(python train_dl.py --new_product 1 | grep "ckpt/" | tail -1)
echo "[Demand] Best checkpoint = $CKPT2"

echo "======================================"
echo "Demand SO-fore (new_product=1) - Forecast"
echo "======================================"
python forecast_dl.py --new_product 1 --ckpt_path $CKPT2



echo "======================================"
echo "All tasks completed!"
echo "======================================"
