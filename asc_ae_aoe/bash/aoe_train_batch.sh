# bash/aoe_train_batch.sh -c 2 -b m1 -m bert-base-uncased

while getopts ':c:m:b:' opt
do
    case $opt in
        c)
        CUDA_IDS="$OPTARG" ;;
        m)
        model_name_or_path="$OPTARG" ;;
        b)
        subname="$OPTARG" ;;
        ?)
        exit 1;;
    esac
done


if [ ! "${subname}" ]
then
  subname="test"
fi


bash/aoe_train.sh -d 14lap -s 40 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14lap -s 50 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14lap -s 60 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14lap -s 70 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14lap -s 80 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}

bash/aoe_train.sh -d 14lap -s 140 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14lap -s 150 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14lap -s 160 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14lap -s 170 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14lap -s 180 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}




bash/aoe_train.sh -d 14res -s 40 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14res -s 50 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14res -s 60 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14res -s 70 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14res -s 80 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}

bash/aoe_train.sh -d 14res -s 140 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14res -s 150 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14res -s 160 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14res -s 170 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}
bash/aoe_train.sh -d 14res -s 180 -c ${CUDA_IDS} -m "${model_name_or_path}" -b ${subname}