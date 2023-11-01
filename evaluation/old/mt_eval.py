from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from evaluation.old.generate import generate_text, LANGS, get_sys_prompt, get_user_prompt
from tqdm import tqdm
import evaluate
import torch
import argparse
import datetime


def extract_prompt(message: str) -> str:
    lines = message.split('Yoruba:')
    return lines[1] if len(lines) > 1 else message

# Shots is a list of dicts with two keys, "input" and "output"
def get_alma_prompt(source_lang: str, target_lang: str, source_sentence: str, shots: list = []) -> str:
    if shots == []:
        return f"Translate this from {source_lang} to {target_lang}:\n{source_lang}: {source_sentence}\n{target_lang}: "
    else:
        resp = f"Examples:\n\n"
        for example in shots:
            resp += f"Translate this from {source_lang} to {target_lang}:\n{source_lang}: {example['input']}\n{target_lang}: {example['output']}\n\n"
        resp += f"Task:\n\nTranslate this from {source_lang} to {target_lang}:\n{source_lang}: {source_sentence}\n{target_lang}: "

        return resp

def get_alma_sys_prompt() -> str:
    return "Follow the instructions and answer to the best of your ability."

def main(model, source_language, target_language, shots):
    s_lang = LANGS[source_language]
    t_lang = LANGS[target_language]

    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    s_data = load_dataset("facebook/flores", f"{s_lang}_Latn")["devtest"]
    t_data = load_dataset("facebook/flores", f"{t_lang}_Latn")["devtest"]
    # system_prompt = f"Translate the following sentence from English to {language}. Provide no justification, please and thank you!"
    system_prompt = get_alma_sys_prompt()
    # instruction = f"Translate the input English sentence to {language}. Provide no justification, please and thank you!"

    examples = []
    if shots > 0:
        s_ex = s_data.select(range(shots))
        t_ex = t_data.select(range(shots))
        s_data = s_data.select(range(shots, len(s_data)))
        t_data = t_data.select(range(shots, len(t_data)))
        for i in range(shots):
            examples.append({"input": s_ex[i]['sentence'], "output": t_ex[i]['sentence']})

    predictions, references = [], []

    for i in tqdm(range(len(s_data))):
        # user_message = f"English: {english[i]['sentence']}"
        user_message = get_alma_prompt(source_language, target_language, s_data[i]['sentence'], examples)

        output = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=300)
        if i < 3:
            print(f"Output {i}: {output}")
        predictions.append(extract_prompt(output))
        references.append([t_data[i]['sentence']])
        # print(f"Llama translation: {output}")
        # print(f"Extracted prompt: {extract_prompt(output)}")
        # print(f"Ground truth: {yoruba[i]['sentence']}")
        # print()

    chrf = evaluate.load("chrf")
    # change word_order=2 for chrf++, subsequent scores will be lower
    # add bleu score
    results_chrf = chrf.compute(predictions=predictions, references=references, word_order=2)
    print(f"Number of Evaluations: {len(s_data)}")
    print(f"ChrF Score: {results_chrf}")
    

if __name__ == "__main__":
    print("-" * 20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model file")
    parser.add_argument("--lang", type=str, help="Language")
    parser.add_argument("--shot", type=int, help="Number of examples to use in n-shot evaluation")
    args = parser.parse_args()

    print(f"Starting {args.shot}-Shot English to {args.lang} Machine Translation Evaluation.\nUsed Model Located At: {args.model}\nStart Time: {str(datetime.datetime.now())}")
    main(model=args.model, source_language="English", target_language=args.lang, shots=args.shot)

    print(f"English to {args.lang} Machine Translation Evaluation Completed. End Time: {str(datetime.datetime.now())}")
    print("-" * 20)

    """
    print(f"Starting {args.shot}-Shot {args.lang} to English Machine Translation Evaluation.\nUsed Model Located At: {args.model}\nStart Time: {str(datetime.datetime.now())}")
    main(model=args.model, source_language=args.lang, target_language="English", shots=args.shot)

    print(f"{args.lang} to English Machine Translation Evaluation Completed. End Time: {str(datetime.datetime.now())}")
    print("-" * 20)
    """
