from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from evaluation.old.generate import generate_text, LANGS, get_sys_prompt, get_user_prompt, SEED
from tqdm import tqdm
from sklearn import metrics
import torch
import argparse
import datetime

def generate_user_text(sample, labels, include_text=True):
    headline = f'Headline: {sample["headline"]}'
    text = f'Text: {sample["text"]}'
    categories = "Categories: "
    for label in labels:
        categories += f'{label}, '
    categories = categories[:-2] + "."

    parts = [headline, text, categories] if include_text else [headline, categories]
    prompt = '\t'.join(parts)
    return prompt


def main(language, model, shots):
    lang = LANGS[language]
    data_path = f"data/{lang}/news/test.tsv"

    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    dataset = load_dataset("csv", data_files={"test": data_path}, delimiter="\t")['test']
    dataset = dataset.shuffle(seed=SEED)

    labels = [
        "entertainment",
        "health",
        "politics",
        "religion",
        "sports"
    ]

    # system_prompt = f"Given a {language} news article with its headline and text, along with a choice of five categories, please provide a one-word response that best categorizes the article. Your input should be the most appropriate category from the provided options.  Please only output one category as your response, thank you!"
    # system_prompt_no_text = f"Given a {language} news article with its headline along with a choice of five categories, please provide a one-word response that best categorizes the article. Your input should be the most appropriate category from the provided options.  Please only output one category as your response, thank you!"
    system_prompt = get_sys_prompt(language, True)
    instruction = f"Given a {language} news article with its headline and text, along with a choice of five categories, please provide a one-word response that best categorizes the article. Your input should be the most appropriate category from the provided options. Please only output one category as your response, thank you!"
    instruction_notext = f"Given a {language} news article with its headline along with a choice of five categories, please provide a one-word response that best categorizes the article. Your input should be the most appropriate category from the provided options.  Please only output one category as your response, thank you!"

    accuracy = [[], 0]
    accuracy_no_text = [[], 0]
    references = []

    examples = []
    if shots > 0:
        dataset_ex = dataset.select(range(shots))
        dataset = dataset.select(range(shots, len(dataset)))
        for i in range(shots):
            examples.append({"input": generate_user_text(dataset_ex[i], labels), "output": dataset_ex[i]['category']})


    for i in tqdm(range(len(dataset))):
        user_message = get_user_prompt(instruction, True, generate_user_text(dataset[i], labels), examples)
        user_message_no_text = get_user_prompt(instruction_notext, True, generate_user_text(dataset[i], labels, False), examples)

        output = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=40)
        if i < 3:
            print(f"Output {i}: {output}")
        
        output_no_text = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message_no_text, max_new_tokens=40)
        if i < 3:
            print(f"Output {i} (No-Text): {output}")

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
    print(f"Number of Evaluations: {len(dataset)}")
    print(f"Accuracy: {score}, confused: {accuracy[1]}")
    print(f"Accuracy (headline only): {score_headline}, confused: {accuracy_no_text[1]}")


if __name__ == "__main__":
    print("-" * 20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model file")
    parser.add_argument("--lang", type=str, help="Language")
    parser.add_argument("--shot", type=int, help="Number of examples to use in n-shot evaluation")
    args = parser.parse_args()

    print(f"Starting {args.shot}-Shot News Classification Evaluation.\nUsed Model Located At: {args.model}\nStart Time: {str(datetime.datetime.now())}")
    main(language=args.lang, model=args.model, shots=args.shot)

    print(f"News Classification Evaluation Completed. End Time: {str(datetime.datetime.now())}")
    print("-" * 20)