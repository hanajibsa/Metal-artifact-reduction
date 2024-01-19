# Metal-artifact-reduction

### Train
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --image_size 256 --exp exp_syndiff --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 4 --contrast1 A --contrast2 B --num_epoch 500 --ngf 64 --embedding_type positional --use_ema --ema_decay 0.999 --r1_gamma 1. --z_emb_dim 256 --lr_d 1e-4 --lr_g 1.6e-4 --lazy_reg 10 --num_process_per_node 2 --save_content --local_rank 0 --input_path /input/path/for/data --output_path /output/for/results
```
### Test 
Test diffusive_module
```
python test_diffusive.py --image_size 256 --exp exp_syndiff --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --embedding_type positional  --z_emb_dim 256 --contrast1 T1  --contrast2 T2 --which_epoch 50 --gpu_chose 0 --input_path /input/path/for/data --output_path /output/for/results
```
Test nondiffusive_module
```
python nondiffusive_test.py --image_size 256 --exp exp_syndiff --num_channels 2 --num_channels_dae 64 --ch_mult 1 1 2 2 4 4 --num_timesteps 4 --num_res_blocks 2 --batch_size 1 --embedding_type positional  --z_emb_dim 256 --contrast1 T1  --contrast2 T2 --which_epoch 50 --gpu_chose 0 --input_path /input/path/for/data --output_path /output/for/results
```
