import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='../data/yor_clean.txt',
    model_prefix='yor_20k',
    model_type="bpe",
    vocab_size=20000
)