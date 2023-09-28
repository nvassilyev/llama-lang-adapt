from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from generate import generate_text
from tqdm import tqdm
from sklearn import metrics
import torch

def generate_user_prompt(sample, labels, include_text=True):
    headline = f'Headline: {sample["headline"]}'
    text = f'Text: {sample["text"]}'
    categories = "Categories: "
    for label in labels:
        categories += f'{label}, '
    categories = categories[:-2] + "."

    parts = [headline, text, categories] if include_text else [headline, categories]
    prompt = '\n'.join(parts)
    return prompt


def main():
    base_model = "../models/llama-2-7b-chat-hf"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )


    dataset = load_dataset("csv", data_files={"test": "test.tsv"}, delimiter="\t")['test']
    dataset = dataset.shuffle(seed=42)

    labels = [
        "entertainment",
        "health",
        "politics",
        "religion",
        "sports"
    ]

    system_prompt = "Given a Yoruba news article with its headline and text, along with a choice of five categories, please provide a one-word response that best categorizes the article. Your input should be the most appropriate category from the provided options.  Please only output one category as your response, thank you!"
    system_prompt_no_text = "Given a Yoruba news article with its headline along with a choice of five categories, please provide a one-word response that best categorizes the article. Your input should be the most appropriate category from the provided options.  Please only output one category as your response, thank you!"

    accuracy = [[], []]
    accuracy_no_text = [[], []]

    for i in tqdm(range(len(dataset))):
        user_message = generate_user_prompt(dataset[i], labels)
        user_message_no_text = generate_user_prompt(dataset[i], labels, False)

        output = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=40)
        
        output_no_text = generate_text(model, tokenizer, system_prompt=system_prompt_no_text,
                message=user_message_no_text, max_new_tokens=40)

        prediction, prediction_no_text = [], []
        for label in labels:
            if output and label in output.lower():
                prediction.append(label)
            if output_no_text and label in output_no_text.lower():
                prediction_no_text.append(label)

        if len(prediction) == 1:
            accuracy[0].append(prediction[0])
            accuracy[1].append(dataset[i]['category'])

        if len(prediction_no_text) == 1:
            accuracy_no_text[0].append(prediction_no_text[0])
            accuracy_no_text[1].append(dataset[i]['category'])
    
    print(f"Accuracy: {metrics.f1_score(accuracy[1], accuracy[0], average='macro')}")
    print(f"Accuracy (headline only): {metrics.f1_score(accuracy_no_text[1], accuracy_no_text[0], average='macro')}")

if __name__ == "__main__":
    main()