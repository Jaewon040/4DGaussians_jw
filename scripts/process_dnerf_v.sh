
export CUDA_VISIBLE_DEVICES=0&&python train.py -s /data/data_4d/dnerf/jumpingjacks --port 7169 --expname "dnerf/jumpingjacks" --configs arguments/dnerf/jumpingjacks.py &
export CUDA_VISIBLE_DEVICES=1&&python train.py -s /data/data_4d/dnerf/trex --port 7170 --expname "dnerf/trex" --configs arguments/dnerf/trex.py 
wait
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/dnerf/jumpingjacks/"  --skip_train --configs arguments/dnerf/jumpingjacks.py &
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/dnerf/trex/"  --skip_train --configs arguments/dnerf/trex.py  
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/dnerf/jumpingjacks/" &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/dnerf/trex/" 

wait
export CUDA_VISIBLE_DEVICES=1&&python train.py -s /data/data_4d/dnerf/mutant --port 7168 --expname "dnerf/mutant" --configs arguments/dnerf/mutant.py &
export CUDA_VISIBLE_DEVICES=0&&python train.py -s /data/data_4d/dnerf/standup --port 7166 --expname "dnerf/standup" --configs arguments/dnerf/standup.py 
wait
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/dnerf/mutant/"  --skip_train --configs arguments/dnerf/mutant.py   &
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/dnerf/standup/"  --skip_train --configs arguments/dnerf/standup.py 
wait
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/dnerf/mutant/"   &
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/dnerf/standup/"  
wait
export CUDA_VISIBLE_DEVICES=1&&python train.py -s /data/data_4d/dnerf/hook --port 7369 --expname "dnerf/hook" --configs arguments/dnerf/hook.py  &
export CUDA_VISIBLE_DEVICES=0&&python train.py -s /data/data_4d/dnerf/hellwarrior --port 7370 --expname "dnerf/hellwarrior" --configs arguments/dnerf/hellwarrior.py 
wait
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/dnerf/hellwarrior/"  --skip_train --configs arguments/dnerf/hellwarrior.py  &
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/dnerf/hook/"  --skip_train --configs arguments/dnerf/hook.py  
wait
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/dnerf/hellwarrior/"  &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/dnerf/hook/" 
wait
export CUDA_VISIBLE_DEVICES=1&&python train.py -s /data/data_4d/dnerf/lego --port 7168 --expname "dnerf/lego" --configs arguments/dnerf/lego.py &
export CUDA_VISIBLE_DEVICES=0&&python train.py -s /data/data_4d/dnerf/bouncingballs --port 7166 --expname "dnerf/bouncingballs" --configs arguments/dnerf/bouncingballs.py 
wait
export CUDA_VISIBLE_DEVICES=1&&python render.py --model_path "output/dnerf/bouncingballs/"  --skip_train --configs arguments/dnerf/bouncingballs.py  &
export CUDA_VISIBLE_DEVICES=0&&python render.py --model_path "output/dnerf/lego/"  --skip_train --configs arguments/dnerf/lego.py  
wait
export CUDA_VISIBLE_DEVICES=1&&python metrics.py --model_path "output/dnerf/bouncingballs/" &
export CUDA_VISIBLE_DEVICES=0&&python metrics.py --model_path "output/dnerf/lego/"   
wait
echo "Done"