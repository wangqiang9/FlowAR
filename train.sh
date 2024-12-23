torchrun --nproc_per_node=1 --master_port 7811 \
main_flowar.py \
--img_size 256 --vae_path /mnt/bn/qihangyu-arnold-dataset-eu/rsc/code/MAR/vae/kl16.ckpt --vae_embed_dim 16 --vae_stride 16 --patch_size 1 \
--model flowar_large --diffloss_d 12 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 5e-5 \
--output_dir ./output_dir/ --resume ./output_dir/ --use_checkpoint \
--data_path /tmp/rsc/DataSet/imagenet/ --cached_path /tmp/rsc/DataSet/imagenet_mar_feature/