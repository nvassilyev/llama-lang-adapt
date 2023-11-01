import collections
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from evaluation.old.generate import generate_text, LANGS, get_sys_prompt, get_user_prompt, SEED, NUM_ROWS
from tqdm import tqdm
import torch
import argparse
import datetime



def span_f1_seqio(targets, predictions):
  """Computes Span based F1 score.

  This function is copied from
  https://github.com/google-research/multilingual-t5/blob/master/multilingual_t5/evaluation/metrics.py

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings

  Returns:
    span f1 across all targets and predictions (Based on CoNLL script)
  """
  true_positives = collections.defaultdict(int)
  false_positives = collections.defaultdict(int)
  false_negatives = collections.defaultdict(int)

  def tags_to_spans(tag_sequence, delimiter=" $$ "):
    """Extract spans from IOB1 or BIO tags."""
    tag_sequence_split = [x.strip() for x in tag_sequence.split(delimiter)]
    tags_entities = []
    for tag_entity in tag_sequence_split:
      tag_entity_split = tag_entity.split(":")
      if len(tag_entity_split) != 2:
        continue
      tag = tag_entity_split[0].strip()
      entity = tag_entity_split[1].strip()
      tags_entities.append((tag, entity))
    return tags_entities

  def compute_f1_metrics(true_positives, false_positives, false_negatives):
    precision = float(true_positives) / float(
        true_positives + false_positives + 1e-13
    )
    recall = float(true_positives) / float(
        true_positives + false_negatives + 1e-13
    )
    f1_measure = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
    return precision, recall, f1_measure

  for target, pred in zip(targets, predictions, strict=True):
    gold_spans = tags_to_spans(target)
    predicted_spans = tags_to_spans(pred)

    for span in predicted_spans:
      if span in gold_spans:
        true_positives[span[0]] += 1
        gold_spans.remove(span)
      else:
        false_positives[span[0]] += 1
    # These spans weren't predicted.
    for span in gold_spans:
      false_negatives[span[0]] += 1

  _, _, f1_measure = compute_f1_metrics(
      sum(true_positives.values()),
      sum(false_positives.values()),
      sum(false_negatives.values()),
  )

  return {"span_f1": f1_measure}


def span_f1(targets: list[str], predictions: list[str]) -> float:
  """Computes span F1 score based on mT5/ByT5 output format."""
  return 100 * span_f1_seqio(targets, predictions)["span_f1"]


def main(model, language, shots):
    lang = LANGS[language]
    data_path = f"data/{lang}/ner/test.jsonl"

    tokenizer = LlamaTokenizer.from_pretrained(model)
    model = LlamaForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    dataset = load_dataset("json", data_files={"test": data_path})['test']
    dataset = dataset.shuffle(seed=SEED)
    dataset = dataset.select(range(NUM_ROWS))
    predictions = []

    # user_message_suffix = f"Named entites refers to names of location (LOC), organization (ORG) and personal name (PER). For example, \'David is an employee of Amazon and he is visiting New York next week to see Esther\' will be PER: David $$ ORG: Amazon $$ LOC: New York $$ PER: Esther\n\nList all the named entities in the passage above written in {language} using $$ as a separator. Note that our given example is in English but you must perform the same on {language} text. Return only the output."
    # system_prompt = "Follow the instructions below and answer to the best of your ability."
    system_prompt = get_sys_prompt(language, True)
    instruction = f"Named entites refers to names of location (LOC), organization (ORG) and personal name (PER). For example, \'David is an employee of Amazon and he is visiting New York next week to see Esther\' will be PER: David $$ ORG: Amazon $$ LOC: New York $$ PER: Esther\n\nList all the named entities in the input passage written in {language} using $$ as a separator."

    examples = []
    if shots > 0:
        dataset_ex = dataset.select(range(shots))
        dataset = dataset.select(range(shots, len(dataset)))
        for i in range(shots):
            examples.append({"input": dataset_ex[i]['text'], "output": dataset_ex[i]['target']})

    targets = [dataset[i]['target'] for i in range(len(dataset))]

    for i in tqdm(range(len(dataset))):

        # user_message = dataset[i]['text'] + "\n\n" + user_message_suffix
        user_message = get_user_prompt(instruction, True, dataset[i]['text'], examples)

        prediction = generate_text(model, tokenizer, system_prompt=system_prompt,
                message=user_message, max_new_tokens=100)
        if i < 3:
            print(f"Output {i}: {prediction}")
        
        predictions.append(prediction)

    print(f"Number of Evaluations: {len(predictions)}")
    print(f"Accuracy: {span_f1(targets, predictions)}")


if __name__ == "__main__":
    print("-" * 20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the model file")
    parser.add_argument("--lang", type=str, help="Language")
    parser.add_argument("--shot", type=int, help="Number of examples to use in n-shot evaluation")
    args = parser.parse_args()

    print(f"Starting {args.shot}-Shot NER Evaluation.\nUsed Model Located At: {args.model}\nStart Time: {str(datetime.datetime.now())}")
    main(model=args.model, language=args.lang, shots=args.shot)

    print(f"NER Evaluation Completed. End Time: {str(datetime.datetime.now())}")
    print("-" * 20)