torchrun --nproc_per_node=8 --master_port=4152 \
eval.py \
--model flowar_huge --diffloss_d 12 --diffloss_w 1536 \
--eval_bsz 128 --num_images 50000 --num_step 50 --cfg 2.8 --guidance 0.7 \
--output_dir /tmp/output_dir \
--resume ../ckpts/FlowAR-H.pth --vae_path ../../MAR/vae/kl16.ckpt \
--data_path /path/to/imagenet1k/ --evaluate



torchrun --nproc_per_node=8 --master_port=4152 \
eval.py \
--model flowar_small --diffloss_d 2 --diffloss_w 1024 \
--eval_bsz 256 --num_images 50000 --num_step 25 --cfg 4.2 --guidance 0.9 \
--output_dir /tmp/output_dir \
--resume ../ckpts/FlowAR-S.pth --vae_path ../../MAR/vae/kl16.ckpt \
--data_path /path/to/imagenet1k/ --evaluate