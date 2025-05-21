
# CUDA_VISIBLE_DEVICES=3 python train_sds.py -s /data/data_4d/dycheck/space-out --port 6087 --expname sds/dycheck/space-out_30000/ --configs arguments/dycheck/default.py --use_sds --sds_start_iter 30000
# wait
# CUDA_VISIBLE_DEVICES=3 python render.py --model_path output/sds/dycheck/space-out_30000/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/space-out
# wait
# CUDA_VISIBLE_DEVICES=3 python metrics.py --model_path output/sds/dycheck/space-out_30000/  

# wait

# CUDA_VISIBLE_DEVICES=3 python train_sds.py -s /data/data_4d/dycheck/space-out --port 6087 --expname sds/dycheck/space-out_5000/ --configs arguments/dycheck/default.py --use_sds --sds_start_iter 5000
# wait
# CUDA_VISIBLE_DEVICES=3 python render.py --model_path output/sds/dycheck/space-out_5000/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/space-out
# wait
# CUDA_VISIBLE_DEVICES=3 python metrics.py --model_path output/sds/dycheck/space-out_5000/  

# wait

# CUDA_VISIBLE_DEVICES=3 python train_sds.py -s /data/data_4d/dycheck/space-out --port 6087 --expname dycheck/space-out_og/ --configs arguments/dycheck/default.py 
# wait
# CUDA_VISIBLE_DEVICES=3 python render.py --model_path output/dycheck/space-out_og/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/space-out
# wait
# CUDA_VISIBLE_DEVICES=3 python metrics.py --model_path output/dycheck/space-out_og/  

# wait

wait
CUDA_VISIBLE_DEVICES=3 python render.py --model_path output/sds/dycheck/spin_5000/ --configs arguments/dycheck/default.py --skip_train --source_path /data/data_4d/dycheck/spin
wait
CUDA_VISIBLE_DEVICES=3 python metrics.py --model_path output/sds/dycheck/spin_5000/  