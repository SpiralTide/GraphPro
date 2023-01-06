CUDA_VISIBLE_DEVICES=0 python Train.py --pretrain_epoch=5   --lr=0.0005
CUDA_VISIBLE_DEVICES=1 python Train.py --pretrain_epoch=10  --lr=0.0005
CUDA_VISIBLE_DEVICES=2 python Train.py --pretrain_epoch=20  --lr=0.0005
CUDA_VISIBLE_DEVICES=3 python Train.py --pretrain_epoch=50  --lr=0.0005
CUDA_VISIBLE_DEVICES=0 python Train.py --pretrain_epoch=100 --lr=0.0005
CUDA_VISIBLE_DEVICES=1 python Train.py --pretrain_epoch=200 --lr=0.0005