for lr in 1e-5
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=3224  --use_env caption_mplug.py \
    --config ./configs/caption_mplug_base.yaml \
    --output_dir output/coco_caption_base_$lr \
    --checkpoint ./mplug_base.pth \
    --text_encoder /nas-alinlp/lcl193798/albef/google/bert-base-uncased \
    --text_decoder /nas-alinlp/lcl193798/albef/google/bert-base-uncased \
    --do_two_optim \
    --lr $lr \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25
done
