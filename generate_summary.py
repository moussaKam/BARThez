import torch
from fairseq.models.bart import BARTModel
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--model_path', type=str)
parse.add_argument('--data_path', type=str)
parse.add_argument('--source_text_path', type=str)
parse.add_argument('--output_path', type=str)
parse.add_argument('--beam', type=int, default=4)
parse.add_argument('--lenpen', type=int, default=1)
parse.add_argument('--sentence_piece_model', type=str, default='sentence_piece_multilingual.model')
parse.add_argument('--max_len_b', type=int, default=200)
parse.add_argument('--min_len', type=int, default=3)
parse.add_argument('--no_repeat_ngram_size', type=int, default=3)

args = parse.parse_args()

bart = BARTModel.from_pretrained(
    '.',
    checkpoint_file=args.model_path,
    data_name_or_path=args.data_path,
    bpe='sentencepiece',
    sentencepiece_vocab=args.sentence_piece_model,
    task='translation',
)


bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32
with open(args.source_text_path) as source, open(args.output_path, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_b=args.max_len_b, min_len=args.min_len, no_repeat_ngram_size=args.no_repeat_ngram_size)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=args.beam, lenpen=args.lenpen, max_len_b=args.max_len_b, min_len=args.min_len, no_repeat_ngram_size=args.no_repeat_ngram_size)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()
