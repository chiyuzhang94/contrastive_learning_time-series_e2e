for task in ECL ETTh2 ETTh1 #ETTh2 ECL 
    do
    for mask in 0.05 0.1 0.2 0.3 
        do
            echo $task$mask
            sbatch --time=48:0:0 --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G --gres=gpu:p100:1 --output=./out_file/"$task"mask"$mask".out --job-name="$task""$mask" "$task"-hyper.sh $mask
            sleep 3
        done
done
