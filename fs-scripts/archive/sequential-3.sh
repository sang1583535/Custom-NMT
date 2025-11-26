#!/bin/bash

ds_src=xstorycloze-id
ds_tgt=xstorycloze-en

for s in eval train; do
    echo "translating $s..."

    for t in input_sentence_1 input_sentence_2 input_sentence_3 input_sentence_4 sentence_quiz1 sentence_quiz2; do
        echo "-- $t"
        fs-scripts/fs-generate.sh experiments/datasets/$ds_src/$s/$t.txt experiments/datasets/$ds_tgt/$s/$t.txt
    done
done

# echo "translating test..."

# echo "--choice1"
# fs-scripts/fs-generate.sh experiments/datasets/xcopa-id/test/choice1.txt experiments/datasets/xcopa-en/test/choice1.txt
# echo

# echo "--choice2"
# fs-scripts/fs-generate.sh experiments/datasets/xcopa-id/test/choice2.txt experiments/datasets/xcopa-en/test/choice2.txt
# echo

# echo "--premise"
# fs-scripts/fs-generate.sh experiments/datasets/xcopa-id/test/premise.txt experiments/datasets/xcopa-en/test/premise.txt
# echo


# echo "translating validation..."

# echo "--choice1"
# fs-scripts/fs-generate.sh experiments/datasets/xcopa-id/validation/choice1.txt experiments/datasets/xcopa-en/validation/choice1.txt
# echo

# echo "--choice2"
# fs-scripts/fs-generate.sh experiments/datasets/xcopa-id/validation/choice2.txt experiments/datasets/xcopa-en/validation/choice2.txt
# echo

# echo "--premise"
# fs-scripts/fs-generate.sh experiments/datasets/xcopa-id/validation/premise.txt experiments/datasets/xcopa-en/validation/premise.txt
# echo