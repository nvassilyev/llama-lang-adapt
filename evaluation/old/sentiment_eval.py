from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from evaluation.old.generate import generate_text, LANGS, get_sys_prompt, get_user_prompt, SEED, NUM_ROWS
from tqdm import tqdm
from sklearn import metrics
import torch
import argparse
import datetime

def main(model, language, shots):
    lang = LANGS[language]
    data_path = f"data/{lang}/sentiment/test.tsv"

    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    dataset = load_dataset("csv", data_files={"test": data_path}, delimiter="\t")['test']
    dataset = dataset.shuffle(seed=SEED)
    dataset = dataset.select(range(NUM_ROWS))

    labels = [
        "positive",
        "negative",
        "neutral"
    ]

    # system_prompt = f"Given a sample text in {language}, along with a choice of three sentiments, please provide a one-word response that best describes the sentiment of the text. Your answer should be the most appropriate sentiment from the provided options. Make sure your answer includes one of these sentiments."
    # system_prompt2 = f"Please analyze the following sample text in {language} and provide a one-word response that best describes the sentiment of the text. Only output the label NEUTRAL, POSITIVE, or NEGATIVE. Do not output anything else."
    system_prompt = get_sys_prompt(language, True)
    instruction = f"Please analyze the following sample text in {language} and provide a one-word response that best describes the sentiment of the text. Only output one of the labels {str(labels)}. Do not output anything else."

    examples = []
    if shots > 0:
        dataset_ex = dataset.select(range(shots))
        dataset = dataset.select(range(shots, len(dataset)))
        for i in range(shots):
            examples.append({"input": dataset_ex[i]['text'], "output": dataset_ex[i]['label']})

    accuracy = [[], []]
    confused_outputs = []

    for i in tqdm(range(len(dataset))):
        # user_message = generate_user_prompt(dataset[i], labels)
        # user_message = f"SAMPLE TEXT: {dataset[i]['text']}\n SENTIMENT PREDICTION:"
        user_message = get_user_prompt(instruction, True, dataset[i]['text'], examples)

        output = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=40)
        if i < 3:
            print(f"Output {i}: {output}")

        prediction = []
        for label in labels:
            if output and label in output.lower():
                prediction.append(label)

        if len(prediction) > 0:
            accuracy[0].append(prediction[0].lower())
        else:
            accuracy[0].append("confused")
            confused_outputs.append(output)

        accuracy[1].append(dataset[i]['label'])

    print(f"Num non-guesses: {len(confused_outputs)}")
    print(f"Total guesses: {len(dataset)}")
    print(f"Accuracy: {metrics.f1_score(accuracy[1], accuracy[0], average='macro')}")

if __name__ == "__main__":
    print("-" * 20)    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model file")
    parser.add_argument("--lang", type=str, help="Language")
    parser.add_argument("--shot", type=int, help="Number of examples to use in n-shot evaluation")
    args = parser.parse_args()

    print(f"Starting {args.shot}-Shot Sentiment Classification Evaluation.\nUsed Model Located At: {args.model}\nStart Time: {str(datetime.datetime.now())}")
    main(model=args.model, language=args.lang, shots=args.shot)

    print(f"Question Sentiment Classifcation Completed. End Time: {str(datetime.datetime.now())}")
    print("-" * 20)
