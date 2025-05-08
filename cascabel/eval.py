import os
import pathlib
from pathlib import Path
import subprocess
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from cascabel.utils import *
from cascabel.constants import *
from spacy.tokens import DocBin, Doc
from spacy.tokens import Doc, Span, Token
from collections import Counter
from spacy.vocab import Vocab
import attr

@attr.define
class Score:
    tp = attr.ib(type=int)
    fp = attr.ib(type=int)
    fn = attr.ib(type=int)
    
    @classmethod
    def from_eval(cls, error_type: str, correct_tokens: List[Token], wrong_tokens: List[Token], gold_span, pred_span, eval_type: str):
        tp = 0
        fn = 0
        fp = 0
        if eval_type == "span_eval":
            if not error_type:
                tp = 1
            elif error_type == "missed":
                fn = 1
            elif error_type == "spurious":
                fp =  1
            else:
                for span in gold_span:
                    fn = fn + 1
                for span in pred_span:
                    fp = fp + 1
            return cls(tp, fp, fn)

        if eval_type == "ts_eval":
            if error_type in ["type", None]:
                tp = tp + len(gold_span[0]) + (len(gold_span[0]) - 1)
            else:
                if "overlap" in error_type:
                    for tok in correct_tokens:
                        tp = tp + 1
                        if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("I"):
                            tp = tp + 1
                        if tok.ent_iob_ == "B" and tok._.iob_pred.startswith("I"):
                            fp = fp + 1
                        if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("B"):
                            fn = fn + 1
                if "missed" in error_type:
                    for tok in wrong_tokens:
                        if tok._.iob_pred == 'O':
                            fn = fn + 1
                            if tok.ent_iob_ == "I":
                                fn = fn + 1
                if "spurious" in error_type:
                    for tok in wrong_tokens:
                        if tok.ent_iob_ == 'O':
                            fp = fp + 1
                            if tok._.iob_pred.startswith("I"):
                                fp = fp + 1
                if error_type == "splitted":
                    for tok in wrong_tokens:
                        if tok._.iob_pred.startswith("B") and tok.ent_iob_ == "I":
                            fn = fn + 1
                            tp = tp + 1 # the token was retrieved, so we add +1 to self.ts_eval.tp, and +1 to self.ts_eval.fn to account for the wrong whitespace
                if error_type == "fused":
                    for i in range(pred_span[0].start, pred_span[0].end):
                        tp = tp + 1 # tokens where retrieved, so we add +1 to self.ts_eval.tp
                        if gold_span[0].sent.doc[i]._.iob_pred.startswith("I") and gold_span[0].sent.doc[i].ent_iob_ == "B":
                            fp = fp + 1
                        elif gold_span[0].sent.doc[i]._.iob_pred.startswith("I") and gold_span[0].sent.doc[i].ent_iob_ == "I":
                            tp = tp + 1 # a non-problematic whitespace was correctly accounted
                if error_type == "missegmented":
                    for span in gold_span:
                        for tok in span:
                            tp = tp + 1
                            if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("I"):
                                tp = tp + 1
                            if tok.ent_iob_ == "B" and tok._.iob_pred.startswith("I"):
                                fp = fp + 1
                            if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("B"):
                                fn = fn + 1

            return cls(tp, fp, fn)

                
                



        
@attr.define
class Eval:
    gold = attr.ib(type=[Span], default=[])
    predicted = attr.ib(type=[Span], default=[])
    sent = attr.ib(type=Span, default=None)
    error = attr.ib(type=bool, default=None)
    error_type = attr.ib(type=str, default=None, validator=attr.validators.in_([None, "missed", "spurious", "overlap", "fused", "type", "splitted", "missegmented", "overlap_spurious_missed", "overlap_spurious", "overlap_missed"]),)
    correct_tokens = attr.ib(type=List[Token], default=[])
    wrong_tokens = attr.ib(type=List[Token], default=[])
    span_eval = attr.ib(type=Score, default=None)
    ts_eval = attr.ib(type=Score, default=None)
    
    


    def equal_to(self, other_eval):
        if self.gold:
            return self.gold.label_ == other_eval.gold.label_ and str(self.gold) == str(other_eval.gold.label_)
        else:
            return self.predicted.label_ == other_eval.predicted.label_ and str(self.predicted) == str(other_eval.predicted.label_)



    def __attrs_post_init__(self):
        self.set_erroneous_tokens()
        self.span_eval = Score.from_eval(self.error_type, self.correct_tokens, self.wrong_tokens, self.gold, self.predicted, "span_eval")
        self.ts_eval = Score.from_eval(self.error_type, self.correct_tokens, self.wrong_tokens, self.gold, self.predicted, "ts_eval")

        
    def set_span_score(self):
        if not self.error:
            self.span_tp = self.span_tp + 1
        elif self.error_type == "missed":
            self.span_fn = self.span_fn + 1
        elif self.error_type == "spurious":
            self.span_fp = self.span_fp + 1
        else:
            for span in self.gold:
                self.span_fn = self.span_fp + 1
            for span in self.predicted:
                self.span_fp = self.span_fp + 1
        

    def set_erroneous_tokens(self):
        if not self.error:
            self.correct_tokens = [token for span in self.gold for token in span]
        elif self.error_type == "missed":
            self.wrong_tokens = [self.sent.doc[i] for i in range(self.gold[0].start, self.gold[0].end)]
            self.correct_tokens = []
        elif self.error_type == "spurious":
            self.wrong_tokens = [self.sent.doc[i] for i in range(self.predicted[0].start, self.predicted[0].end)]
            self.correct_tokens = []
        elif self.error_type == "fused":
            wrong = []
            correct = []

            for span in self.gold:
                for tok in span:
                    if tok.ent_iob_ == "B" and not tok._.iob_pred.startswith("B"): # we only care about the token that should have been B but wasnt B (the token that caused the fusing)
                        wrong.append(tok)
                    else:
                        correct.append(tok)
            self.correct_tokens = correct
            self.wrong_tokens = wrong
        elif self.error_type == "splitted":
            wrong = []
            correct = []
            for i in range(min(self.gold[0].start,self.predicted[0].start), max(self.gold[-1].end,self.predicted[-1].end)):
                for span in self.gold:
                    if span.doc[i].ent_iob_ != "B" and span.doc[i]._.iob_pred.startswith("B"): # we only care about the token that shouldnt have been B but was B (the token that caused the splitting)
                        wrong.append(span.doc[i])
                    else:
                        correct.append(span.doc[i])

            self.correct_tokens = correct
            self.wrong_tokens = wrong
        elif self.error_type == "missegmented":
            gold_tokens = [tok.text for span in self.gold for tok in span]
            self.wrong_tokens = [token for span in self.predicted for token in span if token.text not in gold_tokens]
            self.correct_tokens = [token for span in self.predicted for token in span if token.text in gold_tokens]
        else:
            wrong = []
            correct = []
            for i in range(min(self.gold[0].start,self.predicted[0].start), max(self.gold[-1].end,self.predicted[-1].end)):
                for span in self.gold:
                    if span.doc[i]._.iob_pred != "O" and span.doc[i].ent_iob_ != "O": # we only care about cases where the token should have been O or shouldnt have been O, but we consider a token correct as long as it was labeled as part of an entity (regardless of whether it was B or I)
                        correct.append(span.doc[i])
                    else:
                        wrong.append(span.doc[i])

            self.correct_tokens = correct
            self.wrong_tokens = wrong
            
    def set_span_eval(self):
        if self.error == False:
            self.span_eval["tp"] = self.span_eval["tp"] + 1
        elif self.error_type == "missed":
            self.span_eval["fn"] = self.span_eval["fn"] + 1
        elif self.error_type == "spurious":
            self.span_eval["fp"] = self.span_eval["fp"] + 1
        else:
            for span in self.gold:
                self.span_eval["fn"] = self.span_eval["fn"] + 1
            for span in self.predicted:
                self.span_eval["fp"] = self.span_eval["fp"] + 1
            
    def set_ts_score(self):
        if self.error == False or self.error_type == "type":
            self.ts_eval.tp = self.ts_eval.tp + len(self.gold[0]) + (len(self.gold[0]) - 1)
        else:
            if "overlap" in self.error_type:
                for tok in self.correct_tokens:
                    self.ts_eval.tp = self.ts_eval.tp + 1
                    if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("I"):
                        self.ts_eval.tp = self.ts_eval.tp + 1
                    if tok.ent_iob_ == "B" and tok._.iob_pred.startswith("I"):
                        self.ts_eval.fp = self.ts_eval.fp + 1
                    if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("B"):
                        self.ts_eval.fn = self.ts_eval.fn + 1
            if "missed" in self.error_type:
                for tok in self.wrong_tokens:
                    if tok._.iob_pred == 'O':
                        self.ts_eval.fn = self.ts_eval.fn + 1
                        if tok.ent_iob_ == "I":
                            self.ts_eval.fn = self.ts_eval.fn + 1
            if "spurious" in self.error_type:
                for tok in self.wrong_tokens:
                    if tok.ent_iob_ == 'O':
                        self.ts_eval.fp = self.ts_eval.fp + 1
                        if tok._.iob_pred.startswith("I"):
                            self.ts_eval.fp = self.ts_eval.fp + 1
            if self.error_type == "splitted":
                for tok in self.wrong_tokens:
                    if tok._.iob_pred.startswith("B") and tok.ent_iob_ == "I":
                        self.ts_eval.fn = self.ts_eval.fn + 1
                        self.ts_eval.tp = self.ts_eval.tp + 1 # the token was retrieved, so we add +1 to self.ts_eval.tp, and +1 to self.ts_eval.fn to account for the wrong whitespace
            if self.error_type == "fused":
                for i in range(self.predicted[0].start, self.predicted[0].end):
                    self.ts_eval.tp = self.ts_eval.tp + 1 # tokens where retrieved, so we add +1 to self.ts_eval.tp
                    if self.sent.doc[i]._.iob_pred.startswith("I") and self.sent.doc[i].ent_iob_ == "B":
                        self.ts_eval.fp = self.ts_eval.fp + 1
                    elif self.sent.doc[i]._.iob_pred.startswith("I") and self.sent.doc[i].ent_iob_ == "I":
                        self.ts_eval.tp = self.ts_eval.tp + 1 # a non-problematic whitespace was correctly accounted
            if self.error_type == "missegmented":
                for span in self.gold:
                    for tok in span:
                        self.ts_eval.tp = self.ts_eval.tp + 1
                        if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("I"):
                            self.ts_eval.tp = self.ts_eval.tp + 1
                        if tok.ent_iob_ == "B" and tok._.iob_pred.startswith("I"):
                            self.ts_eval.fp = self.ts_eval.fp + 1
                        if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("B"):
                            self.ts_eval.fn = self.ts_eval.fn + 1

