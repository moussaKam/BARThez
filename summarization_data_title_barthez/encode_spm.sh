#!/bin/bash
SPLITS="train test valid"
TASKS="title"

for TASK in $TASKS
do
    for SPLIT in $SPLITS
    do
        spm_encode --model ../barthez.base/sentence.bpe.model < $SPLIT-article.txt > $SPLIT.spm.article
        spm_encode --model ../barthez.base/sentence.bpe.model < $SPLIT-$TASK.txt > $SPLIT.spm.$TASK
    done
done
