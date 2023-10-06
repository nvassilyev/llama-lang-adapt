from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from generate import generate_text, LANGS, get_sys_prompt, get_user_prompt
from tqdm import tqdm
import evaluate
import torch
import argparse
import datetime

def generate_user_prompt(sample):
    context = f'Context: {sample["text"]}'
    question = f'Question: {sample["question"]}'
    prompt = '\n'.join([context, question])
    return prompt


def main(language, model):
    lang = LANGS[language]
    data_path = f"data/{lang}/qa/test.csv"

    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    dataset = load_dataset("csv", data_files={"test": data_path})['test']

    # system_prompt = "Answer the following question in Yoruba based on the provided context. Provide output only in Yoruba, not in English."
    system_prompt = get_sys_prompt(language, True)
    instruction = f"Answer the following question in {language} based on the provided context. Provide output only in {language}, not in English."

    predictions = []
    references = []

    for i in tqdm(range(len(dataset))):
        user_message = get_user_prompt(instruction, True, generate_user_prompt(dataset[i]))

        output = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=300)
        
        predictions.append({'prediction_text': output, 'id': str(i)})
        references.append({'answers': {'answer_start': [0], 'text': [dataset[i]["answer"]]}, 'id': str(i)})

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=predictions, references=references)
    print(results)

if __name__ == "__main__":
    print("-" * 20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, help="Language")
    parser.add_argument("--model", type=str, help="Path to the model file")
    args = parser.parse_args()

    print(f"Starting Question Answering Evaluation.")
    print(f"Model Located At: {args.model}")
    print(f"Start Time: {str(datetime.datetime.now())}")
    main(language=args.lang, model=args.model)

    print(f"Question Answering Evaluation Completed.\nEnd Time: {str(datetime.datetime.now())}")
    print("-" * 20)
