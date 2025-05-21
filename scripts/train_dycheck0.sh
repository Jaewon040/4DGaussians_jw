exp_name1=$1
data_rootdir=$2

CUDA_VISIBLE_DEVICES=0 python train_sds.py -s /data/data_4d/dycheck/spin --port 6084 --expname sds/dycheck/spin/ --configs arguments/dycheck/default.py --use_sds
wait
CUDA_VISIBLE_DEVICES=0 python render.py --model_path output/$exp_name1/spin/   --configs arguments/$exp_name1/default.py --skip_train --source_path /data/data_4d/dycheck/spin


wait
CUDA_VISIBLE_DEVICES=0 python train.py -s $data_rootdir/dycheck/teddy/ --port 6081 --expname $exp_name1/teddy/ --configs arguments/$exp_name1/default.py 
wait
CUDA_VISIBLE_DEVICES=0 python render.py --model_path output/$exp_name1/teddy/  --skip_train --configs arguments/$exp_name1/default.py 


wait
CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path output/$exp_name1/apple/  
CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path output/sds/dycheck/spin/  
echo "Done"

python render.py --model_path output/sds/dycheck/spin_30000/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/spin


CUDA_VISIBLE_DEVICES=0 python train_sds.py -s /data/data_4d/dycheck/spin --port 6084 --expname sds/dycheck/spin/ --configs arguments/dycheck/default.py --use_sds
wait
CUDA_VISIBLE_DEVICES=0 python render.py --model_path output/sds/dycheck/spin_30000/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/spin
wait
CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path output/sds/dycheck/spin/  
