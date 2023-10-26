from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse
import os

# Adapted from https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/scripts/merge_tokenizer/merge_tokenizers.py

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

def build_tokenizer(llama_tokenizer_dir, new_tokenizer_dir):
    # Load
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
    new_sp_model = spm.SentencePieceProcessor()
    new_sp_model.Load(new_tokenizer_dir)

    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    new_spm = sp_pb2_model.ModelProto()
    new_spm.ParseFromString(new_sp_model.serialized_model_proto())

    print("Llama size: ", len(llama_tokenizer))
    print("New vocab size: ",len(new_sp_model))

    llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
    print(f"Before: {len(llama_spm_tokens_set)}")

    for p in new_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)

    print(f"New model pieces: {len(llama_spm.pieces)}")

    ## Save
    output_sp_dir = 'sp_models'
    output_hf_dir = 'hf_tokenizer' 
    
    with open(output_sp_dir+'/yoruba_llama.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/yoruba_llama.model')

    tokenizer.save_pretrained(output_hf_dir)
    print(f"Yoruba-LLaMA tokenizer has been saved to {output_hf_dir}")
    return output_hf_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama_tokenizer_dir', default="../models/llama-2-7b-chat-hf", type=str)
    parser.add_argument('--new_sp_model_file', default='sp_models/yor_15k.model', type=str)
    args = parser.parse_args()

    output_hf_dir = build_tokenizer(args.llama_tokenizer_dir, args.new_sp_model_file)

    # Test
    llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_tokenizer_dir)
    yoruba_tokenizer = LlamaTokenizer.from_pretrained(args.new_sp_model_file)
    yoruba_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)

    en = "The data that is being collected from them today may be used to judge them in the future and can come to prevent their hopes and dreams."
    yo = "\nDátà tí wọ́n ń gbà lọ́wọ́ wọn lónì lè di lílò láti ṣe ìdájọ́ wọn lọ́jọ́ iwájú ó dẹ̀ lè dèna ìrètí àti àlá wọn."
    print("Test text:\n",en+yo)
    print(f"Tokenized by LLaMA tokenizer:{' '.join(llama_tokenizer.tokenize(en+yo))}")
    print(f"Tokenized by Yoruba tokenizer:{' '.join(yoruba_tokenizer.tokenize(yo))}")
    print(f"Tokenized by Yoruba-LLaMA tokenizer:{' '.join(yoruba_llama_tokenizer.tokenize(en+yo))}")