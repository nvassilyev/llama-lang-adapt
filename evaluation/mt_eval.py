from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from generate import generate_text
from tqdm import tqdm
import evaluate
import torch
import argparse

def extract_prompt(message: str) -> str:
    lines = message.split('Yoruba:')
    return lines[1] if len(lines) > 1 else message

def main(model):
    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
            )


    english = load_dataset("facebook/flores", "eng_Latn")["devtest"]
    yoruba = load_dataset("facebook/flores", "yor_Latn")["devtest"]
    system_prompt = "Translate the following sentence from English to Yoruba. Provide no justification, please and thank you!"

    predictions, references = [], []
    print(len(english))

    for i in tqdm(range(len(english))):
        user_message = f"English: {english[i]['sentence']}"

        output = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=300)
        predictions.append(extract_prompt(output))
        references.append([yoruba[i]['sentence']])
        # print(f"Llama translation: {output}")
        # print(f"Extracted prompt: {extract_prompt(output)}")
        # print(f"Ground truth: {yoruba[i]['sentence']}")
        # print()

    chrf = evaluate.load("chrf")
    results_chrf = chrf.compute(predictions=predictions, references=references)
    print(results_chrf)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model file")
    args = parser.parse_args()
    main(model=args.model)