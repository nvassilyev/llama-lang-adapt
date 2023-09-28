from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from generate import generate_text
from tqdm import tqdm
from sklearn import metrics
import torch

def generate_user_prompt(sample, labels):
    text = f'Sample Text: {sample["text"]}'
    sentiments = "Sentiments: "
    for label in labels:
        sentiments += f'{label}, '
    sentiments = sentiments[:-2] + "."

    parts = [text, sentiments]
    prompt = '\n'.join(parts)
    return prompt


def main():

    ## Here is where you select which model to use
    base_model = "../models/llama-2-7b-chat-hf"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )


    dataset = load_dataset("csv", data_files={"test": "data/sentiment-analysis/test.tsv"}, delimiter="\t")['test']
    dataset = dataset.shuffle(seed=42)

    labels = [
        "positive",
        "negative",
        "neutral"
    ]

    system_prompt = "Given a sample text in Yoruba, along with a choice of three sentiments, please provide a one-word response that best describes the sentiment of the text. Your answer should be the most appropriate sentiment from the provided options. Make sure your answer includes one of these sentiments."

    accuracy = [[], []]
    confused_outputs = []

    for i in tqdm(range(len(dataset))):
        user_message = generate_user_prompt(dataset[i], labels)

        output = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=40)

        prediction = []
        for label in labels:
            if output and label in output.lower():
                prediction.append(label)

        if len(prediction) == 1:
            accuracy[0].append(prediction[0])
        else:
            accuracy[0].append("confused")
            confused_outputs.append(output)

        accuracy[1].append(dataset[i]['label'])

    print(f"Num non-guesses: {len(confused_outputs)}")
    print(f"Confused outputs: {confused_outputs}")
    print(f"Accuracy: {metrics.f1_score(accuracy[1], accuracy[0], average='macro')}")

if __name__ == "__main__":
    main()
