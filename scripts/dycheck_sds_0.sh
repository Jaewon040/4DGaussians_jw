

CUDA_VISIBLE_DEVICES=0 python train_sds.py -s /data/data_4d/dycheck/apple --port 6082 --expname sds/dycheck/apple_30000/ --configs arguments/dycheck/default.py --use_sds --sds_start_iter 30000
wait
CUDA_VISIBLE_DEVICES=0 python render.py --model_path output/sds/dycheck/apple_30000/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/apple
wait
CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path output/sds/dycheck/apple_30000/  

wait

CUDA_VISIBLE_DEVICES=0 python train_sds.py -s /data/data_4d/dycheck/apple --port 6082 --expname sds/dycheck/apple_5000/ --configs arguments/dycheck/default.py --use_sds --sds_start_iter 5000
wait
CUDA_VISIBLE_DEVICES=0 python render.py --model_path output/sds/dycheck/apple_5000/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/apple
wait
CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path output/sds/dycheck/apple_5000/  

wait

CUDA_VISIBLE_DEVICES=0 python train_sds.py -s /data/data_4d/dycheck/apple --port 6082 --expname dycheck/apple_og/ --configs arguments/dycheck/default.py 
wait
CUDA_VISIBLE_DEVICES=0 python render.py --model_path output/dycheck/apple_og/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/apple
wait
CUDA_VISIBLE_DEVICES=0 python metrics.py --model_path output/dycheck/apple_og/  


