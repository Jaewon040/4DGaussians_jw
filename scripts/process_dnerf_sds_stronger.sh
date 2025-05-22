

export CUDA_VISIBLE_DEVICES=0&&python train_sds.py -s /data/data_4d/dnerf/jumpingjacks --port 8269 --expname "sds/dnerf_10000_v2_05w/jumpingjacks" --configs arguments/dnerf/jumpingjacks.py --use_sds --sds_start_iter 10000 --sds_weight 0.5 &
export CUDA_VISIBLE_DEVICES=1&&python train_sds.py -s /data/data_4d/dnerf/trex --port 8280 --expname "sds/dnerf_10000_v2_05w/trex" --configs arguments/dnerf/trex.py --use_sds --sds_start_iter 10000 --sds_weight 0.5
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/sds/dnerf_10000_v2_05w/jumpingjacks/"  --skip_train --configs arguments/dnerf/jumpingjacks.py &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/sds/dnerf_10000_v2_05w/trex/"  --skip_train --configs arguments/dnerf/trex.py  
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/sds/dnerf_10000_v2_05w/jumpingjacks/" &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/sds/dnerf_10000_v2_05w/trex/" 

wait
export CUDA_VISIBLE_DEVICES=0&&python train_sds.py -s /data/data_4d/dnerf/mutant --port 8268 --expname "sds/dnerf_10000_v2_05w/mutant" --configs arguments/dnerf/mutant.py --use_sds --sds_start_iter 10000 --sds_weight 0.5 &
export CUDA_VISIBLE_DEVICES=1&&python train_sds.py -s /data/data_4d/dnerf/standup --port 8266 --expname "sds/dnerf_10000_v2_05w/standup" --configs arguments/dnerf/standup.py --use_sds --sds_start_iter 10000 --sds_weight 0.5

wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/sds/dnerf_10000_v2_05w/mutant/"  --skip_train --configs arguments/dnerf/mutant.py   &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/sds/dnerf_10000_v2_05w/standup/"  --skip_train --configs arguments/dnerf/standup.py 
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/sds/dnerf_10000_v2_05w/mutant/"   &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/sds/dnerf_10000_v2_05w/standup/"  
wait
export CUDA_VISIBLE_DEVICES=0&&python train_sds.py -s /data/data_4d/dnerf/hook --port 8269 --expname "sds/dnerf_10000_v2_05w/hook" --configs arguments/dnerf/hook.py --use_sds --sds_start_iter 10000 --sds_weight 0.5 &
export CUDA_VISIBLE_DEVICES=1&&python train_sds.py -s /data/data_4d/dnerf/hellwarrior --port 8280 --expname "sds/dnerf_10000_v2_05w/hellwarrior" --configs arguments/dnerf/hellwarrior.py --use_sds --sds_start_iter 10000 --sds_weight 0.5
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/sds/dnerf_10000_v2_05w/hellwarrior/"  --skip_train --configs arguments/dnerf/hellwarrior.py  &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/sds/dnerf_10000_v2_05w/hook/"  --skip_train --configs arguments/dnerf/hook.py  
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/sds/dnerf_10000_v2_05w/hellwarrior/"  &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/sds/dnerf_10000_v2_05w/hook/" 
wait
export CUDA_VISIBLE_DEVICES=0&&python train_sds.py -s /data/data_4d/dnerf/lego --port 8268 --expname "sds/dnerf_10000_v2_05w/lego" --configs arguments/dnerf/lego.py --use_sds --sds_start_iter 10000 --sds_weight 0.5 &
export CUDA_VISIBLE_DEVICES=1&&python train_sds.py -s /data/data_4d/dnerf/bouncingballs --port 8266 --expname "sds/dnerf_10000_v2_05w/bouncingballs" --configs arguments/dnerf/bouncingballs.py --use_sds --sds_start_iter 10000 --sds_weight 0.5
wait
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/sds/dnerf_10000_v2_05w/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py  &
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/sds/dnerf_10000_v2_05w/lego/"  --skip_train --configs arguments/dnerf/lego.py  
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/sds/dnerf_10000_v2_05w/bouncingballs/" &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/sds/dnerf_10000_v2_05w/lego/"   
wait
echo "Done"