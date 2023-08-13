for task in ETTm1 ECL
do
	echo $task
	sbatch --time=48:0:0 --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G --gres=gpu:p100:1 --output=./out_file/"$task".out --job-name="$task" "$task".sh
	sleep 1
done
