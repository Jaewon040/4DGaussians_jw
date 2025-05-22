

export CUDA_VISIBLE_DEVICES=3&&python train_sds.py -s /data/data_4d/dnerf/jumpingjacks --port 7269 --expname "sds/dnerf_10000_v2/jumpingjacks" --configs arguments/dnerf/jumpingjacks.py --use_sds --sds_start_iter 10000 &
export CUDA_VISIBLE_DEVICES=2&&python train_sds.py -s /data/data_4d/dnerf/trex --port 7270 --expname "sds/dnerf_10000_v2/trex" --configs arguments/dnerf/trex.py --use_sds --sds_start_iter 10000
wait
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/sds/dnerf_10000_v2/jumpingjacks/"  --skip_train --configs arguments/dnerf/jumpingjacks.py &
export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "output/sds/dnerf_10000_v2/trex/"  --skip_train --configs arguments/dnerf/trex.py  
wait
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/sds/dnerf_10000_v2/jumpingjacks/" &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/sds/dnerf_10000_v2/trex/" 

wait
export CUDA_VISIBLE_DEVICES=2&&python train_sds.py -s /data/data_4d/dnerf/mutant --port 7268 --expname "sds/dnerf_10000_v2/mutant" --configs arguments/dnerf/mutant.py --use_sds --sds_start_iter 10000&
export CUDA_VISIBLE_DEVICES=3&&python train_sds.py -s /data/data_4d/dnerf/standup --port 7266 --expname "sds/dnerf_10000_v2/standup" --configs arguments/dnerf/standup.py --use_sds --sds_start_iter 10000

wait
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/sds/dnerf_10000_v2/mutant/"  --skip_train --configs arguments/dnerf/mutant.py   &
export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "output/sds/dnerf_10000_v2/standup/"  --skip_train --configs arguments/dnerf/standup.py 
wait
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/sds/dnerf_10000_v2/mutant/"   &
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/sds/dnerf_10000_v2/standup/"  
wait
export CUDA_VISIBLE_DEVICES=2&&python train_sds.py -s /data/data_4d/dnerf/hook --port 7269 --expname "sds/dnerf_10000_v2/hook" --configs arguments/dnerf/hook.py --use_sds --sds_start_iter 10000 &
export CUDA_VISIBLE_DEVICES=3&&python train_sds.py -s /data/data_4d/dnerf/hellwarrior --port 7270 --expname "sds/dnerf_10000_v2/hellwarrior" --configs arguments/dnerf/hellwarrior.py --use_sds --sds_start_iter 10000
wait
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/sds/dnerf_10000_v2/hellwarrior/"  --skip_train --configs arguments/dnerf/hellwarrior.py  &
export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "output/sds/dnerf_10000_v2/hook/"  --skip_train --configs arguments/dnerf/hook.py  
wait
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/sds/dnerf_10000_v2/hellwarrior/"  &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/sds/dnerf_10000_v2/hook/" 
wait
export CUDA_VISIBLE_DEVICES=2&&python train_sds.py -s /data/data_4d/dnerf/lego --port 7268 --expname "sds/dnerf_10000_v2/lego" --configs arguments/dnerf/lego.py --use_sds --sds_start_iter 10000&
export CUDA_VISIBLE_DEVICES=3&&python train_sds.py -s /data/data_4d/dnerf/bouncingballs --port 7266 --expname "sds/dnerf_10000_v2/bouncingballs" --configs arguments/dnerf/bouncingballs.py --use_sds --sds_start_iter 10000
wait
export CUDA_VISIBLE_DEVICES=2&&python render.py --model_path "output/sds/dnerf_10000_v2/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py  &
export CUDA_VISIBLE_DEVICES=3&&python render.py --model_path "output/sds/dnerf_10000_v2/lego/"  --skip_train --configs arguments/dnerf/lego.py  
wait
export CUDA_VISIBLE_DEVICES=2&&python metrics.py --model_path "output/sds/dnerf_10000_v2/bouncingballs/" &
export CUDA_VISIBLE_DEVICES=3&&python metrics.py --model_path "output/sds/dnerf_10000_v2/lego/"   
wait
echo "Done"