# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
#<<<<<<< HEAD
from scipy import stats
import numpy as np
#=======

#>>>>>>> 626cb8f7edafab2d19b280ba4c16de4dd6241dc4

@register_criterion('sentence_prediction')
class SentencePredictionCriterion(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'
        logits, _ = model(
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction='sum')
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduction='sum')

        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output['ncorrect'] = (preds == targets).sum()
#<<<<<<< HEAD
            logging_output['TP'] = (torch.logical_and(preds == targets, targets == 1)).sum()
            logging_output['TN'] = (torch.logical_and(preds == targets, targets == 0)).sum()
            logging_output['FP'] = (torch.logical_and(preds != targets, preds == 1)).sum()
            logging_output['FN'] = (torch.logical_and(preds != targets, preds == 0)).sum()
        else:
            logging_output['logits'] = logits.cpu().detach().numpy()
            logging_output['targets'] = targets.cpu().detach().numpy()
#=======

#>>>>>>> 626cb8f7edafab2d19b280ba4c16de4dd6241dc4
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        TP = sum(log.get('TP', 0) for log in logging_outputs)
        TN = sum(log.get('TN', 0) for log in logging_outputs)
        FP = sum(log.get('FP', 0) for log in logging_outputs)
        FN = sum(log.get('FN', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=2)
            metrics.log_scalar('mcc', 100.0 * (TP * TN - FP * FN) / (((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) ** .5), round=2) 
    
        if 'logits' in logging_outputs[0]:
            logits = np.concatenate([log.get('logits') for log in logging_outputs])
            targets = np.concatenate([log.get('targets') for log in logging_outputs])
            spearman_corr = stats.spearmanr(logits, targets).correlation
            metrics.log_scalar('sprcorr', 100.0 * spearman_corr, round=2)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
