#!/bin/bash
cd ..

source ~/ts2vec/bin/activate

mask_rate=$1
losslambda=$2
model=$3
des=$4
### M

python -u main_informer.py --model $model --data ETTm1 --features M --seq_len 672 --label_len 96 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --freq t --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs 50 --patience 5

python -u main_informer.py --model $model --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --freq t --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs 50 --patience 5

python -u main_informer.py --model $model --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --freq t --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs 50 --patience 5

python -u main_informer.py --model $model --data ETTm1 --features M --seq_len 672 --label_len 288 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --freq t --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs 50 --patience 5

python -u main_informer.py --model $model --data ETTm1 --features M --seq_len 672 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --freq t --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs 50 --patience 5

### S

python -u main_informer.py --model $model --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --freq t --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs 50 --patience 5

python -u main_informer.py --model $model --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --freq t --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs 50 --patience 5

python -u main_informer.py --model $model --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --freq t --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs 50 --patience 5

python -u main_informer.py --model $model --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --freq t --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs 50 --patience 5

python -u main_informer.py --model $model --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 3 --batch_size 16 --freq t --mask_rate $mask_rate --des_path $des --loss_lambda $losslambda --train_epochs 50 --patience 5
