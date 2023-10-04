from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import torch

LANGS = {
    "Yoruba": "yor"
}

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_input_token_length(tokenizer: AutoTokenizer, system_prompt: str, message: str) -> int:
    prompt = get_prompt(message, system_prompt)
    input_ids = tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']
    return input_ids.shape[-1]

def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    message: str,
    temperature: float = 0.7,
    repetition_penalty: float = 1.176,
    top_p: float = 0.1,
    top_k: int = 40,
    num_beams: int = 1,
    max_new_tokens: int = 512,
):
    prompt = get_prompt(message, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)

    if input_ids.shape[-1] > 4096:
        print("Input too long, exceeds 4096 tokens")
        return None

    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            # generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split('[/INST]')[1]


def get_prompt(user_message: str, system_prompt: str) -> str:
    return f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]"""
