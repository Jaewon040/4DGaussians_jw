

# CUDA_VISIBLE_DEVICES=2 python train_sds.py -s /data/data_4d/dycheck/teddy --port 6086 --expname sds/dycheck/teddy_30000/ --configs arguments/dycheck/default.py --use_sds --sds_start_iter 30000
# wait
# CUDA_VISIBLE_DEVICES=2 python render.py --model_path output/sds/dycheck/teddy_30000/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/teddy
# wait
# CUDA_VISIBLE_DEVICES=2 python metrics.py --model_path output/sds/dycheck/teddy_30000/  

# wait

# CUDA_VISIBLE_DEVICES=2 python train_sds.py -s /data/data_4d/dycheck/teddy --port 6086 --expname sds/dycheck/teddy_5000/ --configs arguments/dycheck/default.py --use_sds --sds_start_iter 5000
# wait
# CUDA_VISIBLE_DEVICES=2 python render.py --model_path output/sds/dycheck/teddy_5000/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/teddy
# wait
# CUDA_VISIBLE_DEVICES=2 python metrics.py --model_path output/sds/dycheck/teddy_5000/  

# wait

# CUDA_VISIBLE_DEVICES=2 python train_sds.py -s /data/data_4d/dycheck/teddy --port 6086 --expname dycheck/teddy_og/ --configs arguments/dycheck/default.py 
# wait
CUDA_VISIBLE_DEVICES=2 python render.py --model_path output/dycheck/teddy_og/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/teddy
wait
CUDA_VISIBLE_DEVICES=2 python metrics.py --model_path output/dycheck/teddy_og/  
