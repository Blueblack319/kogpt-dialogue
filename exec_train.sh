CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes=1 --nproc_per_node=3 src/main.py --seed=0 --mode="train" --data_dir="data" --train_prefix="train" --valid_prefix="valid" --model_type="gpt2" --bos_token="<bos>" --sp1_token="<sp1>" --sp2_token="<sp2>" --gpu="0" --lr=2e-5 --warmup_ratio=0.0 --batch_size=8 --num_workers=0 --num_epochs=10 --max_len=1024 --max_turns=5 --ckpt_dir="saved_models"