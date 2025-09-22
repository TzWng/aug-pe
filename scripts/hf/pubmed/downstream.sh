result_folder="/content/drive/MyDrive/SecPE/pubmed_train/augpe_2p/new"

export WANDB_DISABLED="true"


max_seq_length=512
batch_size=32
min_token_threshold=50
lr=3e-4
wd=0.01
cluster="augpe"
rp="2p"
item=${result_folder}




for model in 'bert-small' 
do
num_train_epochs=10
for  (( iter=${num_train_epochs}; iter>=0; iter-- ))
do
train_file="${item}/${cluster}_${rp}.csv"
echo $train_file
if [ -e "$train_file" ]; then
    echo "$train_file does exist."

    output_dir=${result_folder}/ep${num_train_epochs}_${model}_${cluster}_${rp}/

    if [ -e "$output_dir/eval_results.json" ]; then
        echo "$output_dir/eval_results.json does exist. -- SKIP running classification"
    else

    python utility_eval/run_clm.py \
        --model_name_or_path prajjwal1/${model} \
        --clean_dataset  --min_token_threshold ${min_token_threshold} \
        --output_dir ${output_dir} \
        --train_file ${train_file} \
        --validation_file data/pubmed/dev.csv \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size ${batch_size} \
        --learning_rate ${lr} \
        --do_eval \
        --do_train \
        --weight_decay ${wd} \
        --num_train_epochs ${num_train_epochs} \
        --save_total_limit 2 \
        --overwrite_output_dir --overwrite_cache 

    python utility_eval/run_clm.py \
        --model_name_or_path prajjwal1/${model} \
        --output_dir ${output_dir} \
        --train_file ${train_file}\
        --validation_file data/pubmed/test.csv \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size ${batch_size} \
        --learning_rate ${lr} \
        --do_eval \
        --do_train \
        --weight_decay ${wd} \
        --num_train_epochs 0 
    fi
fi
done
done

for model in 'bert-mini' 
do
num_train_epochs=20
for  (( iter=epochs; iter>=0; iter-- ))
do
train_file="${item}/${cluster}_${rp}.csv"
echo $train_file
if [ -e "$train_file" ]; then
    echo "$train_file does exist."

    output_dir=${result_folder}/ep${num_train_epochs}_${model}_${cluster}_${rp}/

    if [ -e "$output_dir/eval_results.json" ]; then
        echo "$output_dir/eval_results.json does exist. -- SKIP running classification"
    else

    python utility_eval/run_clm.py \
        --model_name_or_path prajjwal1/${model} \
        --clean_dataset  --min_token_threshold ${min_token_threshold} \
        --output_dir ${output_dir} \
        --train_file ${train_file}\
        --validation_file data/pubmed/dev.csv \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size ${batch_size} \
        --learning_rate ${lr} \
        --do_eval \
        --do_train \
        --weight_decay ${wd} \
        --num_train_epochs ${num_train_epochs} \
        --save_total_limit 2 \
        --overwrite_output_dir --overwrite_cache 

    python utility_eval/run_clm.py \
        --model_name_or_path prajjwal1/${model} \
        --output_dir ${output_dir} \
        --train_file ${train_file} \
        --validation_file data/pubmed/test.csv \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size ${batch_size} \
        --learning_rate ${lr} \
        --do_eval \
        --do_train \
        --weight_decay ${wd} \
        --num_train_epochs 0 
    fi
fi
done
done





