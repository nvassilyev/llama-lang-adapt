from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch import no_grad
from tqdm import tqdm
import json
import time

# https://github.com/tatsu-lab/stanford_alpaca

checkpoint = 'facebook/nllb-200-3.3B'
device = 'cuda'
target_language = 'yor_Latn'

def translate(text, tokenizer, model):
    with no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated_tokens = model.generate(
            **inputs.to(device), forced_bos_token_id=tokenizer.lang_code_to_id[target_language], max_length=512
        )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


if __name__ == '__main__':
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    with open('alpaca_data.json', 'r') as file:
        alpaca_dataset = json.load(file)

    start = time.perf_counter()

    translated = []
    for i in tqdm(range(len(alpaca_dataset))):
        translated.append(
            {
                'instruction': translate(alpaca_dataset[i]['instruction'], tokenizer, model),
                'input': translate(alpaca_dataset[i]['input'], tokenizer, model) if alpaca_dataset[i]['input'] else "",
                'output': translate(alpaca_dataset[i]['output'], tokenizer, model),
            }
        )
    with open(f"alpaca_data_{target_language}.json", "w", encoding="utf-8") as f:
        json.dump(translated, f, indent=4, ensure_ascii=False)
    
    end = time.perf_counter() - start
    print(f"Finished in {end} seconds")
