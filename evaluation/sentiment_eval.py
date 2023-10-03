from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from generate import generate_text
from tqdm import tqdm
from sklearn import metrics
import torch
import argparse

def generate_user_prompt(sample, labels):
    text = f'Sample Text: {sample["text"]}'
    sentiments = "Sentiments: "
    for label in labels:
        sentiments += f'{label}, '
    sentiments = sentiments[:-2] + "."

    parts = [text, sentiments]
    prompt = '\n'.join(parts)
    return prompt


def main(model, data, language="Yoruba"):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
            )


    dataset = load_dataset("csv", data_files={"test": data}, delimiter="\t")['test']
    dataset = dataset.shuffle(seed=42)

    labels = [
        "positive",
        "negative",
        "neutral"
    ]

    system_prompt = f"Given a sample text in {language}, along with a choice of three sentiments, please provide a one-word response that best describes the sentiment of the text. Your answer should be the most appropriate sentiment from the provided options. Make sure your answer includes one of these sentiments."
    system_prompt2 = f"Please analyze the following sample text in {language} and provide a one-word response that best describes the sentiment of the text. Only output the label NEUTRAL, POSITIVE, or NEGATIVE. Do not output anything else."

    accuracy = [[], []]
    confused_outputs = []

    for i in tqdm(range(10)):
        # user_message = generate_user_prompt(dataset[i], labels)
        user_message = f"SAMPLE TEXT: {dataset[i]['text']}\n SENTIMENT PREDICTION:"

        output = generate_text(model, tokenizer, system_prompt=system_prompt2,
                message=user_message, max_new_tokens=40)

        prediction = []
        for label in labels:
            if output and label in output.lower():
                prediction.append(label)

        if len(prediction) > 0:
            accuracy[0].append(prediction[0].lower())
        else:
            accuracy[0].append("confused")
            confused_outputs.append(output)
            print(user_message)
            print(output)

        accuracy[1].append(dataset[i]['label'])

    print(f"Num non-guesses: {len(confused_outputs)}")
    print(f"Confused outputs: {confused_outputs}")
    print(f"Accuracy: {metrics.f1_score(accuracy[1], accuracy[0], average='macro')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the data file")
    parser.add_argument("--model", type=str, help="Path to the model file")
    args = parser.parse_args()
    main(data=args.data, model=args.model)
