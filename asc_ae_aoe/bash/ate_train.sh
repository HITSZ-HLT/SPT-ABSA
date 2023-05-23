while getopts ':d:s:c:m:b:p:o:' opt
do
    case $opt in
        d)
        dataset="$OPTARG" ;;
        s)
        seed="$OPTARG" ;;
        c)
        CUDA_IDS="$OPTARG" ;;
        m)
        model_name_or_path="$OPTARG" ;;
        b)
        subname="$OPTARG" ;;
        p)
        training_data_prop="$OPTARG" ;;
        o)
        output_dir="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done



if [ ! "${CUDA_IDS}" ]
then
  CUDA_IDS=0,1,2,3
fi


if [ ! "${dataset}" ]
then
  dataset="14res"
fi


if [ ! "${seed}" ]
then
  seed=42
fi


if [ ! "${training_data_prop}" ]
then
  training_data_prop=1
fi


if [ ! "${model_name_or_path}" ]
then
  model_name_or_path="bert-base-uncased"
fi


if [ ! "${subname}" ]
then
  subname="test"
fi


if [ ! "${output_dir}" ]
then
  output_dir="/data/zhangyice/2022/sentiment pre-training/"
fi

echo ${model_name_or_path}


max_seq_length=-1
gradient_clip_val=1
warmup_steps=0
weight_decay=0.01

precision=16
batch_size=24
learning_rate=2
max_epochs=10

data_dir="data"
output_dir="${output_dir}/downstream-tasks/ate/output/"


CUDA_VISIBLE_DEVICES=${CUDA_IDS} python ate_train.py \
  --gpus=1 \
  --precision=${precision} \
  --data_dir "${data_dir}" \
  --model_name_or_path "${model_name_or_path}" \
  --output_dir "${output_dir}" \
  --learning_rate ${learning_rate}e-5 \
  --train_batch_size ${batch_size} \
  --eval_batch_size ${batch_size} \
  --seed $seed \
  --warmup_steps ${warmup_steps} \
  --gradient_clip_val ${gradient_clip_val} \
  --weight_decay ${weight_decay} \
  --max_seq_length ${max_seq_length} \
  --max_epochs ${max_epochs} \
  --do_train \
  --output_sub_dir ${subname} \
  --dataset $dataset \
  --training_data_prop ${training_data_prop}



rm -r "${output_dir}/dataset=${dataset},seed=${seed},m=${subname}"