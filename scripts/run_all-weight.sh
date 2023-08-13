weight=0.5
model=informer
mask_rate=0.3
des="./hyper_results"
train_epoch=20
patience=5
l2norm=false
moco_average_pool=false

for task in ETTh1 ETTh2 ETTm1 ECL 
    do
    for weight in 0.01 0.05 0.1 0.3 0.5 0.7
        do
            echo $task$weight
            sbatch --time=48:0:0 --account=rrg-mageed --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G --gres=gpu:1 --output=./out_file/"$task""$weight".out --job-name="$task""$weight" "$task"-hyper.sh $mask_rate $weight $model $des $train_epoch $patience $l2norm $moco_average_pool
            sleep 1
        done
done
