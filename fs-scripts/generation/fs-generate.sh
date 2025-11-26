#!/bin/bash

src=zh
tgt=en
fixed=en-zh

checkpoint_path=fs-checkpoints/zh-en/checkpoint_best.pt
src_path=$1
tgt_path=$2

lang=$src-$tgt

orig=fs-data/$fixed
prep=$orig/prep
bpe_code=$prep/code

fs_data=fs-data-bin/$fixed

LIB=lib
SCRIPTS=$LIB/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
FSROOT=$LIB/fairseq/fairseq_cli

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "generating with fairseq..."

# cat $src_path | \
# fairseq-interactive $fs_data \
#     --path $checkpoint_path \
#     --beam 5 --source-lang $src --target-lang $tgt \
#     --user-dir $LIB/rdrop \
#     --buffer-size 2500 --max-tokens 20000 \
#     --tokenizer moses \
#     --remove-bpe --bpe subword_nmt --bpe-codes $bpe_code > $tgt_path.generate
# grep ^D- $tgt_path.generate | sort -V | cut -f3 | perl $DETOKENIZER -threads 8 -l $tgt > $tgt_path

# for zh-en
cat $src_path | \
fairseq-interactive $fs_data \
    --path $checkpoint_path \
    --beam 5 --source-lang $src --target-lang $tgt \
    --user-dir $LIB/rdrop \
    --buffer-size 2500 --max-tokens 20000 \
    --remove-bpe --bpe subword_nmt --bpe-codes $bpe_code > $tgt_path.generate
grep ^D- $tgt_path.generate | sort -V | cut -f3 | perl $DETOKENIZER -threads 8 -l $tgt > $tgt_path

echo "generation completed"