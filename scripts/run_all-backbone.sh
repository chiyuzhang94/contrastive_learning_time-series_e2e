weight=0.0
model=tcn
mask_rate=0.3
des="./backbone_10l"
train_epoch=30
patience=5
l2norm="True"
moco_average_pool="False"
data_aug="cost"
cos_lr="True"
e_layers=10
learning_rate=0.001
for task in ETTh1 ETTh2 ETTm11 ETTm12 ECL 
    do
    for model in 'dtcn' #'dtcn-moco' "cost-e2e"
        do
            echo $task$model
            sbatch --time=120:0:0 --output=./out_file/"$task""$model"_l2"$l2norm"_avg${moco_average_pool}_data${data_aug}_cos${cos_lr}.out --job-name="$task""$model" "$task"-hyper.sh $mask_rate $weight $model $des $train_epoch $patience $l2norm $moco_average_pool $data_aug $cos_lr $e_layers $learning_rate
            sleep 10
        done
done
