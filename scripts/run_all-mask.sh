weight=0.5
model=informer
mask_rate=0.3
des="./hyper_results"
for task in ETTh2 #ECL 
    do
    for mask_rate in 0.05 #0.1 0.2 0.3 0.4 0.5
        do
            echo $task$mask_rate
            sbatch --time=48:0:0 --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G --gres=gpu:1 --output=./out_file/"$task""$mask_rate".out --job-name="$task""$mask_rate" "$task"-hyper.sh $mask_rate $weight $model $des
            sleep 1
        done
done
