DICT=../barthez.base/dict.txt
SRC=article
TGT=title
TRAIN=train.spm
VALID=valid.spm
TEST=test.spm
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${TRAIN} \
  --validpref ${VALID} \
  --testpref ${TEST} \
  --destdir data-bin \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 8 
