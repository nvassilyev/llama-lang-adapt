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

def main(model, language, shots):
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
    system_prompt = get_sys_prompt(language, True, shots)
    instruction = f"Translate the input English sentence to {language}. Provide no justification, please and thank you!"

    examples = []
    if shots > 0:
        english_ex = english.select(range(shots))
        yoruba_ex = yoruba.select(range(shots))
        english = english.select(range(shots, len(english)))
        yoruba = yoruba.select(range(shots, len(yoruba)))
        for i in range(shots):
            examples.append({"input": english_ex[i]['sentence'], "output": yoruba_ex[i]['sentence']})

    predictions, references = [], []

    for i in tqdm(range(len(english))):
        # user_message = f"English: {english[i]['sentence']}"
        user_message = get_user_prompt(instruction, True, english[i]['sentence'], examples)

        output = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=300)
        if i < 3:
            print(f"Output {i}: {output}")
        predictions.append(extract_prompt(output))
        references.append([yoruba[i]['sentence']])
        # print(f"Llama translation: {output}")
        # print(f"Extracted prompt: {extract_prompt(output)}")
        # print(f"Ground truth: {yoruba[i]['sentence']}")
        # print()

    chrf = evaluate.load("chrf")
    # change word_order=2 for chrf++, subsequent scores will be lower
    # add bleu score
    results_chrf = chrf.compute(predictions=predictions, references=references, word_order=2)
    print(f"Number of Evaluations: {len(english)}")
    print(f"ChrF Score: {results_chrf}")
    

if __name__ == "__main__":
    print("-" * 20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model file")
    parser.add_argument("--lang", type=str, help="Language")
    parser.add_argument("--shot", type=int, help="Number of examples to use in n-shot evaluation")
    args = parser.parse_args()

    print(f"Starting {args.shot}-Shot Machine Translation Evaluation.\nUsed Model Located At: {args.model}\nStart Time: {str(datetime.datetime.now())}")
    
    main(model=args.model, language=args.lang, shots=args.shot)

    print(f"Machine Translation Evaluation Completed. End Time: {str(datetime.datetime.now())}")
    print("-" * 20)
