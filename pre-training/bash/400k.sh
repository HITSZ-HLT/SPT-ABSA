# bash/400k.sh -c 2,3 -d {your_data_dir} -o {your_output_dir} -c {your_cache_dir}

while getopts ':c:d:o:' opt
do
    case $opt in
        c)
        CUDA_IDS="$OPTARG" ;;
        d)
        base_data_dir="$OPTARG" ;;
        o)
        output_data_dir="$OPTARG" ;;
        e)
        cache_dir="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done


if [ ! "${CUDA_IDS}" ]
then
  CUDA_IDS=0,1,2,3
fi


gradient_clip_val=0
# seem to large, maybe 8 is suitable
learning_rate=10
max_seq_length=128
warmup_steps=0
weight_decay=0.01
precision=16

model_name_or_path="bert-base-uncased"

train_batch_size=256
eval_batch_size=256
# base_data_dir="/data/zhangyice/2022/sentiment pre-training/8m28d/annotated/lite_12m25d"
data_dir="yelp__amazon/Cell_Phones_and_Accessories__amazon/All_Beauty__amazon/AMAZON_FASHION__amazon/Appliances__amazon/Arts_Crafts_and_Sewing__amazon/Automotive__amazon/Books__amazon/CDs_and_Vinyl__amazon/Clothing_Shoes_and_Jewelry__amazon/Digital_Music__amazon/Electronics__amazon/Gift_Cards__amazon/Grocery_and_Gourmet_Food__amazon/Home_and_Kitchen__amazon/Industrial_and_Scientific__amazon/Kindle_Store__amazon/Luxury_Beauty__amazon/Magazine_Subscriptions__amazon/Movies_and_TV__amazon/Musical_Instruments__amazon/Office_Products__amazon/Patio_Lawn_and_Garden__amazon/Pet_Supplies__amazon/Prime_Pantry__amazon/Sports_and_Outdoors__amazon/Toys_and_Games__amazon/Video_Games"
# cache_dir="/data/zhangyice/.cache/huggingface/datasets/400k_lr10/"
# output_dir="/data/zhangyice/2022/sentiment pre-training/output_model/2023/400k_lr10"



CUDA_VISIBLE_DEVICES=${CUDA_IDS} python sentiment_pretrain.py \
  --gpus=2 \
  --accelerator='ddp' \
  --precision=${precision} \
  --base_data_dir "${base_data_dir}" \
  --data_dir "${data_dir}" \
  --cache_dir "${cache_dir}" \
  --model_name_or_path "${model_name_or_path}" \
  --output_dir "${output_dir}" \
  --learning_rate ${learning_rate}e-5 \
  --train_batch_size ${train_batch_size} \
  --eval_batch_size ${eval_batch_size} \
  --seed 42 \
  --warmup_steps ${warmup_steps} \
  --gradient_clip_val ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_seq_length ${max_seq_length} \
  --max_steps 400_000 \
  --val_check_interval 5_000 \
  --accumulate_grad_batches 2 \
  --num_workers 12 \
  --train_size 81_920_000 \
  --test_size 102_400
