cls_batch_size=32
result_folder="/content/drive/MyDrive/SecPE/results-yelp-random-all"

### calculate acc 
# bash scripts/hf/yelp/downstream.sh
num_train_epochs=5
for seed in 42 4 6 8
do
for label in "label2"
do
for  (( iter=${num_train_epochs}; iter>=0; iter-- ))
do
for file_name in "train.csv"
do
for folder in "qwen_800_10" "qwen_800_50" "qwen_600_0"
do
train_file="${result_folder}/${folder}/${file_name}"
if [ -e "$train_file" ]; then
    echo "$train_file does exist."

    output_dir=${result_folder}/${folder}/${label}_seed${seed}_${file_name%_prefixed.csv}/
    if [ -e "${output_dir}test_${num_train_epochs}.0_results.json" ]; then
        echo "${output_dir}test_${num_train_epochs}.0_results.json  does exist. -- SKIP running classification"
    else
        echo "${output_dir}test_${num_train_epochs}.0_results.json  does not exist. -- RUN running classification"
        python utility_eval/run_classification.py \
            --report_to none --clean_dataset --model_name_or_path  roberta-base \
            --output_dir ${output_dir} \
            --train_file ${train_file} --validation_file data/yelp/dev.csv --test_file data/yelp/test.csv \
            --do_train --do_eval --do_predict --max_seq_length 512 --per_device_train_batch_size ${cls_batch_size} --per_device_eval_batch_size ${cls_batch_size} \
            --learning_rate 3e-5 --num_train_epochs ${num_train_epochs} \
            --overwrite_output_dir --overwrite_cache \
            --save_strategy epoch --save_total_limit 2 --load_best_model_at_end \
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
done

