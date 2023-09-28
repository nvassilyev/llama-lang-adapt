from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from generate import generate_text
from tqdm import tqdm
import evaluate
import torch

def extract_prompt(message: str) -> str:
    lines = message.split('\n')
    return lines[1] if len(lines) >= 1 else message


def main():
    base_model = "../models/llama-2-7b-chat-hf"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
            )


    english = load_dataset("facebook/flores", "eng_Latn")["devtest"]
    yoruba = load_dataset("facebook/flores", "yor_Latn")["devtest"]
    system_prompt = "Translate the following sentence from English to Yoruba. Provide no justification, please and thank you!"

    predictions, references = [], []

    for i in tqdm(range(len(english))):
        user_message = f"English: {english[i]['sentence']}"

        output = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=512)
        predictions.append(output)
        references.append([yoruba[i]['sentence']])

    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=prediction, references=references)
    

if __name__ == "__main__":
    main()