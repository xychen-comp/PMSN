python -m torch.distributed.launch --nproc_per_node=8 --use_env train_PMSN.py --cos_lr_T 320 --model sew_resnet34 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epochs 320 --data-path /datasets/imagenet --tet

python -m torch.distributed.launch --nproc_per_node=8 --use_env train_PMSN.py --cos_lr_T 320 --model sew_resnet18 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epochs 320 --data-path /datasets/imagenet --tet -b 64
