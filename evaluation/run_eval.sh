#!/bin/bash

# Dictionary of languages and their short forms. Ensure an equivalent dictionary LANGS exists in generate.py.
declare -A language_list=( ["Yoruba"]="yor")

# Edit the models you wish to test. Should be a relative directory pointing to your model.
# models=(
#     "/mnt/disk/llama-lang-adapt/models/llama-2-7b-hf"
#     "/mnt/disk/llama-lang-adapt/models/llama-2-7b-chat-hf"
# )

models=(
    "/mnt/disk/llama-lang-adapt/models/llama-2-7b-hf"
    "/mnt/disk/llama-lang-adapt/models/llm-africa/alpaca-v2/yor-inst-prompt"
    "/mnt/disk/llama-lang-adapt/models/llm-africa/alpaca-v2/yor-inst-prompt-mt"
)

# Edit the tasks you wish to test. Available tasks are: ner, mt, news, qa, sentiment.
tasks=(
    "ner"
    "news"
    "sentiment"
)

# Edit which versions of n-shot learning you wish to perform
shots=(
    "0"
    "1"
    "3"
    "5"
)

# Evaluates all tasks on all models in the list above. Outputs are stored in the results directory.
function eval () {

    lang=${language_list[$1]}

    for model in "${models[@]}"; do
        for task in "${tasks[@]}"; do
            for shot in "${shots[@]}"; do
                echo "Starting model $model with $shot shots on task $task"
                python ${task}_eval.py --model $model --lang $1 --shot $shot >> results/${lang}/${task}.txt
                echo "Model $model with $shot shots completed on task $task"
            done
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
