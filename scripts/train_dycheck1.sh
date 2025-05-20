exp_name1=$1
data_rootdir=$2
CUDA_VISIBLE_DEVICES=1 python train.py -s $data_rootdir/dycheck/space-out --port 6083 --expname $exp_name1/space-out/ --configs arguments/$exp_name1/default.py 
wait
CUDA_VISIBLE_DEVICES=1 python render.py --model_path output/$exp_name1/space-out/   --configs arguments/$exp_name1/default.py 


wait
CUDA_VISIBLE_DEVICES=1 python train.py -s $data_rootdir/dycheck/apple/ --port 6082 --expname $exp_name1/apple/ --configs arguments/$exp_name1/default.py 
wait
CUDA_VISIBLE_DEVICES=1 python render.py --model_path output/$exp_name1/apple/  --skip_train --configs arguments/$exp_name1/default.py 


wait
CUDA_VISIBLE_DEVICES=1 python metrics.py --model_path output/$exp_name1/teddy/  
wait
CUDA_VISIBLE_DEVICES=1 python metrics.py --model_path output/$exp_name1/space-out/  
echo "Done"