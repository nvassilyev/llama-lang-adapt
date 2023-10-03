from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from generate import generate_text
from tqdm import tqdm
from sklearn import metrics
import torch
import argparse

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


def main(data, model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    dataset = load_dataset("csv", data_files={"test": data}, delimiter="\t")['test']
    dataset = dataset.shuffle(seed=42)

    labels = [
        "entertainment",
        "health",
        "politics",
        "religion",
        "sports"
    ]
    language = "Yoruba"

    system_prompt = f"Given a {language} news article with its headline and text, along with a choice of five categories, please provide a one-word response that best categorizes the article. Your input should be the most appropriate category from the provided options.  Please only output one category as your response, thank you!"
    system_prompt_no_text = f"Given a {language} news article with its headline along with a choice of five categories, please provide a one-word response that best categorizes the article. Your input should be the most appropriate category from the provided options.  Please only output one category as your response, thank you!"

    accuracy = [[], 0]
    accuracy_no_text = [[], 0]
    references = []

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

        if len(prediction) > 0:
            accuracy[0].append(prediction[0])
        else:
            accuracy[0].append("confused")
            accuracy[1] += 1
            # print(output)

        if len(prediction_no_text) > 0:
            accuracy_no_text[0].append(prediction_no_text[0])
        else:
            accuracy_no_text[0].append("confused")
            accuracy_no_text[1] += 1
            # print(output_no_text)
        
        references.append(dataset[i]['category'])
    
    score = metrics.f1_score(references, accuracy[0], average='macro')
    score_headline = metrics.f1_score(references, accuracy_no_text[0], average='macro')
    print(f"Accuracy: {score}, confused: {accuracy[1]}")
    print(f"Accuracy (headline only): {score_headline}, confused: {accuracy_no_text[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the data file")
    parser.add_argument("--model", type=str, help="Path to the model file")
    args = parser.parse_args()
    main(data=args.data, model=args.model)