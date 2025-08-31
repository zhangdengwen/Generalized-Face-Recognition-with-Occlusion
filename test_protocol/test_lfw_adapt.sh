#!/bin/bash
export PYTHONPATH=/home/srp/face_recognition:$PYTHONPATH
cd /home/srp/face_recognition/test_protocol

python /home/srp/face_recognition/test_protocol/test_lfw_adapt.py \
    --test_set 'CALFW_LFW' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:0' \
    --adapt_path "/home/srp/face_recognition/out_dir/Epoch_19_adapt.pt" \
    --model_path "/home/srp/face_recognition/out_dir/Epoch_19.pt"

python /home/srp/face_recognition/test_protocol/test_lfw_adapt.py \
    --test_set 'LFW' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:0' \
    --adapt_path "/home/srp/face_recognition/out_dir/Epoch_19_adapt.pt" \
    --model_path "/home/srp/face_recognition/out_dir/Epoch_19.pt"



python /home/srp/face_recognition/test_protocol/test_lfw_adapt.py \
    --test_set 'MEGLASS' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:0' \
    --adapt_path "/home/srp/face_recognition/out_dir/Epoch_19_adapt.pt" \
    --model_path "/home/srp/face_recognition/out_dir/Epoch_19.pt"

python /home/srp/face_recognition/test_protocol/test_lfw_adapt.py \
    --test_set 'CALFW_CROP' \
    --data_conf_file 'data_conf.yaml' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file 'backbone_conf.yaml' \
    --batch_size 128 \
    --device 'cuda:0' \
    --adapt_path "/home/srp/face_recognition/out_dir/Epoch_19_adapt.pt" \
    --model_path "/home/srp/face_recognition/out_dir/Epoch_19.pt"

#  python /home/srp/face_recognition/test_protocol/test_lfw_adapt.py \
#      --test_set 'CPLFW' \
#      --data_conf_file 'data_conf.yaml' \
#      --backbone_type 'MobileFaceNet' \
#      --backbone_conf_file 'backbone_conf.yaml' \
#      --batch_size 128 \
#      --device 'cuda:0' \
#      --adapt_path "/home/srp/face_recognition/out_dir/Epoch_19_adapt.pt" \
#      --model_path "/home/srp/face_recognition/out_dir/Epoch_19.pt"

#  conda activate occlusion_face
#  chmod +x /home/srp/face_recognition/test_protocol/test_lfw_adapt.sh
#  bash /home/srp/face_recognition/test_protocol/test_lfw_adapt.sh



# 运行基线测试集，需要下载测试集，放入test_data文件夹，
# 然后修改test_protocol里面的文件路径，修改test_lfw_adapt.sh里面路径，正常运行就是基线结果。
# 运行完基线测试集后，首先把测试集改成有crop的

# 首先需要更新face_sdk文件夹下的core文件夹，原项目复制不全
# 更新完后需要在test_protocol.lfw.face_cropper.crop_lfw_by_arcface.py修改项目路径
# 修改原代码中的system.append，运行该程序生成crop_lfw测试集。
# 需要注意cplfw,calfw这两个数据集AMOFR给的不对，要用faceX_zero的数据集，
# 里面的路径和导入都不对都要改，最后路径也要改,melass这个数据集没有对应的，搞不定。
# 最后要改test_protocol.data_conf.yaml,里面所有路径都要改才能运行