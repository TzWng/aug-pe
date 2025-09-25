cls_batch_size=32
result_folder="/content/drive/MyDrive/SecPE/synthetic_text/openreview_qwen1.5b"


num_train_epochs=10
max_seq_length=512
model="roberta-base"
min_token_threshold=100
item=${result_folder}

# "augpe_infty_prefixed.csv" "cluster15_infty_prefixed.csv" "cluster20_infty_prefixed.csv" "cluster25_infty_prefixed.csv"
## calculate acc 
for seed in 42 666 777
do
for label in "label1"
do
for  (( iter=${num_train_epochs}; iter>=0; iter-- ))
do
for file_name in "cluster20_50p_prefixed.csv" "cluster20_2p_prefixed.csv"
do
train_file="${result_folder}/${file_name}"
echo $train_file
if [ -e "$train_file" ]; then
    echo "$train_file does exist."
    output_dir=${result_folder}/${label}_seed${seed}_${file_name}/
    if [ -e "${output_dir}test_${iter}.0_results.json" ]; then
        echo "${output_dir}test_${iter}.0_results.json  does exist. -- SKIP running classification"
    else
        echo "${output_dir}test_${iter}.0_results.json  does not exist. -- RUN running classification"
        python utility_eval/run_classification.py \
            --report_to none --clean_dataset  --min_token_threshold ${min_token_threshold} \
            --model_name_or_path  ${model} \
            --output_dir ${output_dir} \
            --train_file ${train_file} \
            --validation_file data/openreview/iclr23_reviews_val.csv \
            --test_file data/openreview/iclr23_reviews_test.csv \
            --do_train --do_eval --do_predict --max_seq_length ${max_seq_length} --per_device_train_batch_size ${cls_batch_size} --per_device_eval_batch_size ${cls_batch_size} \
            --learning_rate 3e-5 --num_train_epochs ${iter} \
            --overwrite_output_dir --overwrite_cache \
            --save_strategy epoch --save_total_limit 1 --load_best_model_at_end \
            --logging_strategy epoch \
            --seed ${seed} \
            --metric_for_best_model accuracy_all --greater_is_better True \
            --evaluation_strategy epoch --label_column_name ${label}
    fi
else
    echo "$train_file does not exist."
fi
done
done
done
done

