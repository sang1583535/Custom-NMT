#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

# step to run
pick=0
src=ta
tgt=en
src2=tam
tgt2=eng
fixed=en-ta
custom=ta
while getopts p:s:t:r: flag
do
    case "${flag}" in
		p) pick=${OPTARG};;
		# s) src=${OPTARG};;
		# t) tgt=${OPTARG};;
    esac
done

is_run() {
	local step=$1
	if [ $pick == $step ]; then
		return 1
	else
		return 0
	fi
}

LIB=lib
SCRIPTS=$LIB/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
TOKENIZER_CUSTOM=$LIB/custom-tokenizers/$custom/tokenize.py
DETOKENIZER_CUSTOM=$LIB/custom-tokenizers/$custom/detokenize.py
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
APEXROOT=$LIB/apex
FSROOT=$LIB/fairseq/fairseq_cli
BPEROOT=$LIB/subword-nmt/subword_nmt
BPE_TOKENS=36000

is_run 1
if [ $? -eq 1 ]; then
    cd $LIB

    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git

    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone https://github.com/rsennrich/subword-nmt.git

    cd ..
fi

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

lang=$src-$tgt
orig=fs-data/$fixed
prep=$orig/prep
tmp=$prep/tmp
tst=datasets/flores101/devtest

mkdir -p $tmp $prep

is_run 2
if [ $? -eq 1 ]; then
    echo "pre-processing train data..."
    for l in $src $tgt; do
        rm -f $tmp/train.$l
        if [ "$l" == "$custom" ]; then
            cat $orig/train.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                python3.10 $TOKENIZER_CUSTOM >> $tmp/train.$l
        else
            cat $orig/train.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.$l
        fi
    done

    echo "pre-processing val data..."
    for l in $src $tgt; do
        rm -f $tmp/val.$l
        if [ "$l" == "$custom" ]; then
            cat $orig/val.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                python3.10 $TOKENIZER_CUSTOM >> $tmp/valid.$l
        else
            cat $orig/val.$l | \
                perl $NORM_PUNC $l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $TOKENIZER -threads 8 -a -l $l >> $tmp/valid.$l
        fi
    done
fi

is_run 3
if [ $? -eq 1 ]; then
    echo "pre-processing test data..."
    for l in $src $tgt; do
        if [ "$l" == "$src" ]; then
            t="$src2"
        else
            t="$tgt2"
        fi
        echo $tst/$t.devtest
        if [ "$l" == "$custom" ]; then
            grep '' $tst/$t.devtest | \
                # sed -e 's/<seg id="[0-9]*">\s*//g' | \
                # sed -e 's/\s*<\/seg>\s*//g' | \
                sed -e "s/\’/\'/g" | \
                python3.10 $TOKENIZER_CUSTOM > $tmp/test.$l
        else
            grep '' $tst/$t.devtest | \
                # sed -e 's/<seg id="[0-9]*">\s*//g' | \
                # sed -e 's/\s*<\/seg>\s*//g' | \
                sed -e "s/\’/\'/g" | \
                perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
        fi
        echo ""
    done
fi

TRAIN=$tmp/train.$lang
BPE_CODE=$prep/code

is_run 5
if [ $? -eq 1 ]; then
    rm -f $TRAIN
    # combine parallel texts
    paste "$tmp/train.$src" "$tmp/train.$tgt" > "$TRAIN"
fi

is_run 6
if [ $? -eq 1 ]; then
    echo "learn_bpe.py on ${TRAIN}..."
    rm -f $BPE_CODE
    python3.10 $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
fi

is_run 7
if [ $? -eq 1 ]; then
    for l in $src $tgt; do
        for f in train.$l valid.$l test.$l; do
            echo "apply_bpe.py to ${f}..."
            python3.10 $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
        done
    done
fi

is_run 8
if [ $? -eq 1 ]; then
    perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
    perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250
fi

is_run 9
if [ $? -eq 1 ]; then
    for L in $src $tgt; do
        cp $tmp/bpe.test.$L $prep/test.$L
    done
fi

fs_task=$lang
fs_data=fs-data-bin/$fixed
fs_chkpts=fs-checkpoints/$fs_task
fs_res=fs-results/$fs_task
fs_log=logs/$fs_task

mkdir -p $fs_log

is_run 10
if [ $? -eq 1 ]; then
    echo "preprocessing with fairseq..."
    rm -rf $fs_data
    python3.10 $FSROOT/preprocess.py --source-lang $src --target-lang $tgt \
        --trainpref $prep/train --validpref $prep/valid --testpref $prep/test \
        --destdir $fs_data --workers 20
fi

is_run 11
if [ $? -eq 1 ]; then
    echo "training with fairseq..."

    bleu_args='{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}'

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export PYTHONPATH=$(pwd)/$LIB/fairseq:$PYTHONPATH
    alias python=python3.10
    alias python3=python3.10
    cmd="python3.10 $FSROOT/train.py $fs_data
        --arch transformer_wmt23_hw_tsc
        --task rdrop_translation --source-lang $src --target-lang $tgt
        --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08
        --lr-scheduler inverse_sqrt --lr 0.0005
        --warmup-init-lr 1e-07 --warmup-updates 4000
        --weight-decay 0.0001 --clip-norm 0.0
        --reg-alpha 5
        --label-smoothing 0.1 --criterion reg_label_smoothed_cross_entropy
        --max-tokens 8192 --update-freq 6 --max-epoch 8 --fp16
        --skip-invalid-size-inputs-valid-test
        --eval-bleu --eval-bleu-args '$bleu_args'
        --eval-bleu-detok moses --eval-bleu-remove-bpe
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
        --user-dir $LIB/rdrop
        --save-dir $fs_chkpts
        --patience 3"
    cmd="nohup "${cmd}" > $fs_log/train.log &"
    eval $cmd
    echo $! > $fs_log/train.pid
    tail -f $fs_log/train.log
fi

is_run 12
if [ $? -eq 1 ]; then
    echo "generating with fairseq..."
    mkdir -p $fs_res/tmp
    # export CUDA_VISIBLE_DEVICES=0,1,2,3

    if [ "$src" == "$custom" ]; then
        grep '' $tst/$src2.devtest | \
            python3.10 $TOKENIZER_CUSTOM > $fs_res/tmp/test.$src.tok
    else
        grep '' $tst/$src2.devtest | \
            perl $TOKENIZER -threads 8 -a -l $src > $fs_res/tmp/test.$src.tok
    fi

    checkpoints=("1" "2" "3" "4" "5" "6" "7" "8" "_best")
    for c in "${checkpoints[@]}";
    do
        echo "evaluating checkpoint-$c..."

        if [ "$src" == "$custom" ]; then
            cat $fs_res/tmp/test.$src.tok | \
            python3.10 $FSROOT/interactive.py $fs_data \
                --path $fs_chkpts/checkpoint$c.pt \
                --beam 5 --source-lang $src --target-lang $tgt \
                --user-dir $LIB/rdrop \
                --remove-bpe --bpe subword_nmt --bpe-codes $BPE_CODE > $fs_res/generate-test-$c.txt
        else
            cat $tst/$src2.devtest | \
            python3.10 $FSROOT/interactive.py $fs_data \
                --path $fs_chkpts/checkpoint$c.pt \
                --beam 5 --source-lang $src --target-lang $tgt \
                --user-dir $LIB/rdrop --tokenizer moses \
                --remove-bpe --bpe subword_nmt --bpe-codes $BPE_CODE > $fs_res/generate-test-$c.txt
        fi

        if [ "$tgt" == "$custom" ]; then
            grep ^D- $fs_res/generate-test-$c.txt | sort -V | cut -f3 | \
                python3.10 $DETOKENIZER_CUSTOM > $fs_res/translations-$c.txt
        else
            grep ^D- $fs_res/generate-test-$c.txt | sort -V | cut -f3 | \
                perl $DETOKENIZER -threads 8 -l $tgt > $fs_res/translations-$c.txt
        fi

        echo "checkpoint-$c evaluation completed"
        sleep 10
    done
fi

is_run 20
if [ $? -eq 1 ]; then
    echo "computing evalution scores..."

    alias python=python3.10
    alias python3=python3.10
    checkpoints=("1" "2" "3" "4" "5" "6" "7" "8" "_best")
    for c in "${checkpoints[@]}";
    do
        bleu=$(cat "$fs_res/translations-$c.txt" | sacrebleu "$tst/$tgt2.devtest" -b -l "$lang" -m bleu)
        chrf=$(cat "$fs_res/translations-$c.txt" | sacrebleu "$tst/$tgt2.devtest" -b -l "$lang" -m chrf --chrf-word-order 2)
        cmet=$(comet-score -s "$tst/$src2.devtest" -r "$tst/$tgt2.devtest" -t "$fs_res/translations-$c.txt" --quiet --only_system | awk -F 'score: ' '{print $2}')
        echo "----------"
        echo "$lang $src2-$tgt2 (checkpoint-$c) -- BLEU: $bleu; chrF++: $chrf; COMET-22: $cmet"
        echo "----------"
    done
fi