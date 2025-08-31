#!/bin/bash
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1

# 创建日志文件夹（如果不存在）
mkdir -p log

python training_mode/conventional_training/train_webface_kd_adapt.py \
    --data_root /home/srp/face_recognition/data/CASIA-WebFace \
    --train_file /home/srp/face_recognition/training_mode/webface_train_list.txt \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file './training_mode/backbone_conf.yaml' \
    --head_type 'ArcFace' \
    --head_conf_file './training_mode/head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir' \
    --epoches 20 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 48 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'mv-hrnet' \
    --device 'cuda' \
    --w 100 \
    --teacher_pth '/home/srp/face_recognition/pretrained_models/'

    2>&1 | tee log/log_sunglass_glass.log
#    --resume \
#    --pretrain_model "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir17/Epoch_4.pt" \
#    --pretrain_adapt "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir17/Epoch_4_adapt.pt" \
#    --pretrain_header "/CIS20/lyx/FaceX-Zoo-main-new/training_mode/conventional_training/out_dir17/Epoch_4_header.pt" \

# tmux new -s zdw
# tmux attach -t zdw
# tmux kill-session -t zdw

#  conda activate occlusion_face
#  chmod +x training_mode/conventional_training/train.sh
#  bash training_mode/conventional_training/train.sh
