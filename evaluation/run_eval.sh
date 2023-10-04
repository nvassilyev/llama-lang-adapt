#!/bin/bash

# Dictionary of languages and their short forms. Ensure an equivalent dictionary LANGS exists in generate.py.
declare -A language_list=( ["Yoruba"]="yor")

# Edit the models you wish to test. Should be a relative directory pointing to your model.
models=(
    "/mnt/disk/llama-lang-adapt/models/llama-2-7b-chat-hf"
)

# Edit the tasks you wish to test. Available tasks are: ner, mt, news, qa, sentiment.
tasks=(
    "ner"
    "mt"
    "news"
    "qa"
    "sentiment"
)

# Evaluates all tasks on all models in the list above. Outputs are stored in the results directory.
function eval () {

    lang=${language_list[$1]}

    for model in "${models[@]}"; do
        for task in "${tasks[@]}"; do
            echo "Starting model $model on task $task"
            python ${task}_eval.py --model $model --lang $1 >> results/${lang}/${task}.txt
            echo "Model $model completed on task $task"
        done
    done
}

while getopts l: flag 
do
    case "${flag}" in
        l) language=${OPTARG};;
    esac
done

if [ -z "$language" ]
then
      echo "Must provide testing language. For example, bash run_eval.sh -l Yoruba"
else
    eval $language
fi
