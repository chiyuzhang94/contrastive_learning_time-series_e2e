weight=0.5
mask_rate=0.3
des="./backbone_10l"
train_epoch=20
patience=5
l2norm="True"
moco_average_pool="False"
data_aug="False"
cos_lr="False"
e_layers=5
learning_rate=0.001
mare="False"
time_feat="False"
closs_decay="False"
for task in ETTh2 ETTh11 ETTh12 ETTm11 ETTm12 ECL #ETTh1 ETTh2 ETTm11 ETTm12 ECL 
    do
    for model in 'lstm-moco'
        do
            echo $task$model
            sbatch --time=120:0:0 --output=./out_file/${task}${model}_l2n${l2norm}_avg${moco_average_pool}_weight${weight}_layer${e_layers}_data${data_aug}_cos${cos_lr}_timeF${time_feat}_mare${mare}_wdecay${closs_decay}.out --job-name="$task""$model" "$task"-hyper.sh $mask_rate $weight $model $des $train_epoch $patience $l2norm $moco_average_pool $data_aug $cos_lr $e_layers $learning_rate $mare $time_feat $closs_decay
            sleep 10
        done
done
