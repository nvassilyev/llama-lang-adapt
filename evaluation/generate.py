from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import torch

LANGS = {
    "Yoruba": "yor"
}
SEED = 42
NUM_ROWS = 400

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_sys_prompt(language: str, has_input: bool, n_shot: int = 0) -> str:
    if has_input:
        if n_shot == 0:
            return f"Below is an instruction that describes a task in English, paired with an input in {language} that provides further context. Write a response that appropriately completes the request."
        else:
            return f"Below is an instruction that describes a task in English, paired with an input in {language} that provides further context. You are provided also with {n_shot} example inputs and outputs to make the problem clearer. Write a response that appropriately completes the request."
    else:
        return f"Below is an instruction that describes a task in English. Write a response that appropriately completes the request."

def get_user_prompt(instruction: str, has_input: bool = False, input_str: str = "", shot_list: list(dict(str, str)) = []) -> str:
    ### shot_list is a list of examples with two keys, "input" and "output"
    if has_input:
        if len(shot_list) == 0:
            return f" ### Instruction:\n{instruction}\n\n### Input:\n{input_str}\n\n### Response: "
        else:
            resp = f" ### Instruction:\n{instruction}\n\n###Examples:\n"
            for (i, example) in enumerate(shot_list):
                resp += f"{str(i)}. #Input: {example['input']}\t #Output: {example['output']}\n"
            resp += f"\n### Input:\n{input_str}\n\n### Response: "
            return resp
    else:
        return f" ### Instruction:\n{instruction}\n\n### Response: "
    
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
