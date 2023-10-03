from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from generate import generate_text
from tqdm import tqdm
import evaluate
import torch
import argparse

def generate_user_prompt(sample):
    context = f'Context: {sample["text"]}'
    question = f'Question: {sample["question"]}'
    prompt = '\n\n'.join([context, question])
    return prompt


def main(data, model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    dataset = load_dataset("csv", data_files={"test": data})['test']

    system_prompt = "Answer the following question in Yoruba based on the provided context. Provide output only in Yoruba, not in English."

    predictions = []
    references = []

    for i in tqdm(range(len(dataset))):
        user_message = generate_user_prompt(dataset[i])

        output = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=300)
        
        predictions.append({'prediction_text': output, 'id': str(i)})
        references.append({'answers': {'answer_start': [0], 'text': [dataset[i]["answer"]]}, 'id': str(i)})

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=predictions, references=references)
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the data file")
    parser.add_argument("--model", type=str, help="Path to the model file")
    args = parser.parse_args()
    main(data=args.data, model=args.model)
