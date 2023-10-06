from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from generate import generate_text, LANGS, get_sys_prompt, get_user_prompt
from tqdm import tqdm
import evaluate
import torch
import argparse
import datetime

def extract_prompt(message: str) -> str:
    lines = message.split('Yoruba:')
    return lines[1] if len(lines) > 1 else message

def main(model, language):
    lang = LANGS[language]

    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    english = load_dataset("facebook/flores", "eng_Latn")["devtest"]
    yoruba = load_dataset("facebook/flores", f"{lang}_Latn")["devtest"]
    # system_prompt = f"Translate the following sentence from English to {language}. Provide no justification, please and thank you!"
    system_prompt = get_sys_prompt(language, True)
    instruction = f"Translate the input English sentence to {language}. Provide no justification, please and thank you!"

    predictions, references = [], []

    for i in tqdm(range(len(english))):
        # user_message = f"English: {english[i]['sentence']}"
        user_message = get_user_prompt(instruction, True, english[i]['sentence'])

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
    print("-" * 20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model file")
    parser.add_argument("--lang", type=str, help="Language")
    args = parser.parse_args()

    print(f"Starting Machine Translation Evaluation.\nUsed Model Located At: {args.model}\nStart Time: {str(datetime.datetime.now())}")
    
    main(model=args.model, language=args.lang)

    print(f"Machine Translation Evaluation Completed. End Time: {str(datetime.datetime.now())}")
    print("-" * 20)
