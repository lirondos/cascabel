import pathlib
from collections import defaultdict
from cascabel.utils import *
from cascabel.constants import *
from spacy.tokens import DocBin, Doc
from spacy.tokens import Doc, Span, Token
from collections import Counter


from spacy.vocab import Vocab
import attr

from cascabel.eval import Eval

if os.name == "nt":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

@attr.define
class Cascabel:
    test = attr.ib(type=Doc, default=None)
    training = attr.ib(type=Doc, default=None)
    lexicon = attr.ib(type=set, default=set())
    gazetteer = attr.ib(type=set, default=set())
    prediction = attr.ib(type=Doc, default=None)
    previously_seen_tokens = attr.ib(type=Dict[str, Set[str]], init=False, default=[])
    previously_seen_ents = attr.ib(type=Dict[str, Set[str]], init=False, default=[])
    evaluation = attr.ib(type=List[Eval], init=False, default=[])

    @prediction.validator
    def sanity_check_prediction(self, attribute, value):
        if value:
            if len(value) != len(self.test):
                raise ValueError("Prediction and test must have same length")
            if str(value) != str(self.test):
                raise ValueError("Prediction and test must have same text")

    def __attrs_post_init__(self):
        self.set_extensions(self.test)
        if self.training:
            self.set_extensions(self.training)
            self.previously_seen_tokens = self.set_previously_seen_tokens()
            self.previously_seen_ents = self.set_previously_seen_ents()
            self.set_training_extensions(self.test)
        if self.gazetteer:
            set_lexicon_extension()
            self.update_outsourced_extensions(self.test)
        if self.lexicon:
            set_lexicon_extension()
            self.update_outsourced_extensions(self.test)
        if self.prediction:
            self.add_prediction_layer()
            self.set_eval_instances()
            self.set_extensions(self.prediction)
            if self.training:
                self.set_training_extensions(self.prediction)


    def evaluate_token_classification(self):
        evaluate = defaultdict(lambda: defaultdict(int))
        for tok in self.test:
            if tok._.iob_pred == get_full_bio_tag(tok):
                evaluate[get_full_bio_tag(tok)]["tp"] = evaluate[get_full_bio_tag(tok)]["tp"] + 1
            else:
                evaluate[get_full_bio_tag(tok)]["fn"] = evaluate[get_full_bio_tag(tok)]["fn"] + 1
                evaluate[tok._.iob_pred]["fp"] = evaluate[tok._.iob_pred]["fp"] + 1
        fp =  0
        fn = 0
        tp = 0
        for tag, mydict in evaluate.items():
            if tag=="O":
                continue
            fp = fp + mydict["fp"]
            fn = fn + mydict["fn"]
            tp = tp + mydict["tp"]

        return get_p_r_f(fp, fn, tp)

    def evaluate_span_classification(self):
        fp =  0
        fn = 0
        tp = 0
        for eval in self.evaluation:
            if eval.error == False:
                tp = tp + 1
            elif "missed" == eval.error_type:
                fn = fn + 1
            else:
                fp = fp + 1
        return get_p_r_f(fp, fn, tp)







    def get_true_positives(self, evals: List[Eval]):
        return len([eval for eval in evals if eval.error==False])


    def get_false_positives(self, evals: List[Eval]):
        return len([span for eval in evals for span in eval.predicted if eval.error == True and eval.error_type != "missed"])

    def get_false_negatives(self, evals: List[Eval]):
        return len([span for eval in evals for span in eval.gold if eval.error == True and eval.error_type != "spurious"])


    def span_p_r_f1(self, evals: List[Eval]):
        tp = len([eval for eval in evals if eval.error==False])
        fp = len([span for eval in evals for span in eval.predicted if eval.error == True and eval.error_type != "missed"])
        fn = len([span for eval in evals for span in eval.gold if eval.error == True and eval.error_type != "spurious"])

        return p_r_f1(tp, fp, fn)




    def get_precision(self, evals: List[Eval]):
        try:
            return self.get_true_positives(evals) / (self.get_true_positives(evals)+ self.get_false_positives(evals))
        except ZeroDivisionError:
            return 0

    def get_recall(self, evals: List[Eval]):
        try:
            return self.get_true_positives(evals) / (self.get_true_positives(evals) + self.get_false_negatives(evals))
        except ZeroDivisionError:
            return 0

    def get_error_types(self, evals: List[Eval]):
        return Counter([eval.error_type for eval in evals])


    def token_separator_p_r_f(self, evals: List[Eval]):
        tp = 0
        fp = 0
        fn = 0
        for eval in evals:
            if eval.error == False or eval.error_type == "type":
                    tp = tp + len(eval.gold[0]) + (len(eval.gold[0]) - 1)
            else:
                if "overlap" in eval.error_type:
                    for tok in eval.correct_tokens:
                        tp = tp + 1
                        if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("I"):
                            tp = tp + 1
                        if tok.ent_iob_ == "B" and tok._.iob_pred.startswith("I"):
                            fp = fp + 1
                        if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("B"):
                            fn = fn + 1
                if "missed" in eval.error_type:
                    for tok in eval.wrong_tokens:
                        if tok._.iob_pred == 'O':
                            fn = fn + 1
                            if tok.ent_iob_ == "I":
                                fn = fn + 1
                if "spurious" in eval.error_type:
                    for tok in eval.wrong_tokens:
                        if tok.ent_iob_ == 'O':
                            fp = fp + 1
                            if tok._.iob_pred.startswith("I"):
                                fp = fp + 1
                if eval.error_type == "splitted":
                    for tok in eval.wrong_tokens:
                        if tok._.iob_pred.startswith("B") and tok.ent_iob_ == "I":
                            fn = fn + 1
                            tp = tp + 1 # the token was retrieved, so we add +1 to TP, and +1 to FN to account for the wrong whitespace
                if eval.error_type == "fused":
                    for i in range(eval.predicted[0].start, eval.predicted[0].end):
                        tp = tp + 1 # tokens where retrieved, so we add +1 to TP
                        if eval.sent.doc[i]._.iob_pred.startswith("I") and eval.sent.doc[i].ent_iob_ == "B":
                            fp = fp + 1
                        elif eval.sent.doc[i]._.iob_pred.startswith("I") and eval.sent.doc[i].ent_iob_ == "I":
                            tp = tp + 1 # a non-problematic whitespace was correctly accounted
                if eval.error_type == "missegmented":
                    for span in eval.gold:
                        for tok in span:
                            tp = tp + 1
                            if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("I"):
                                tp = tp + 1
                            if tok.ent_iob_ == "B" and tok._.iob_pred.startswith("I"):
                                fp = fp + 1
                            if tok.ent_iob_ == "I" and tok._.iob_pred.startswith("B"):
                                fn = fn + 1
        return p_r_f1(tp, fp, fn)







    def add_prediction_layer(self):
        Doc.set_extension("ents_pred", default=None, force=True)
        Span.set_extension("ents_pred", default=None, force=True)
        Token.set_extension("iob_pred", default=None, force=True)
        self.test._.ents_pred = self.prediction.ents
        for sent_gold, sent_pred in zip(self.test.sents, self.prediction.sents):
            sent_gold._.ents_pred = sent_pred.ents
        for tok_gold, tok_pred in zip(self.test, self.prediction):
            tok_gold._.iob_pred = get_full_bio_tag(tok_pred)

    def get_retrieval_capacity(self):
        return len([tok for tok in self.test if tok._.iob_pred != "O" and tok.ent_iob_ !="O"])/len([tok for tok in self.test if tok.ent_iob_ !="O"])

    def get_noise_level(self):
        return len([tok for tok in self.test if tok._.iob_pred != "O" and tok.ent_iob_=="O"])/len([tok for tok in self.test if tok.ent_iob_=="O"])

    def get_segmentation_error_rate(self):
        return len([eval for eval in self.evaluation if self.is_segmentation_error(eval)])/len(self.test.ents)

    def get_segmentation_success_rate(self):
        return len([eval for eval in self.evaluation if eval.gold and eval.gold._.has_adjacent and eval.error==False])/len([eval for eval in self.evaluation if eval.gold and eval.gold._.has_adjacent])


    def is_segmentation_error(self, eval: Eval):
        return eval.error and ("missegmented" in eval.error_type or "splitted" in eval.error_type or "fused" in eval.error_type)

    def get_number_of_errors(self):
        return len([eval for eval in self.evaluation if eval.error])


    def set_eval_instances(self):
        eval_list = []
        for sent in self.test.sents:
            gold_i = 0
            pred_i = 0
            while gold_i < len(sent.ents) and pred_i < len(sent._.ents_pred):
                ent_gold: Span = sent.ents[gold_i]
                ent_pred: Span = sent._.ents_pred[pred_i]
                if is_same_ent(ent_gold, ent_pred):
                    eval = Eval([ent_gold], [ent_pred], sent, False, None)
                    eval_list.append(eval)
                    gold_i = gold_i + 1
                    pred_i = pred_i + 1
                elif is_type_swap(ent_gold, ent_pred):
                    eval = Eval([ent_gold], [ent_pred], sent, True, "type")
                    eval_list.append(eval)
                    gold_i = gold_i + 1
                    pred_i = pred_i + 1
                elif gold_i + 1 != len(sent.ents) and is_fused_ent(ent_gold, sent.ents[gold_i + 1], ent_pred):
                    eval = Eval([ent_gold, sent.ents[gold_i + 1]], [ent_pred], sent, True, "fused")
                    eval_list.append(eval)
                    gold_i = gold_i + 2
                    pred_i = pred_i + 1
                elif pred_i + 1 != len(sent._.ents_pred) and is_splitted_ent(ent_gold, ent_pred, sent._.ents_pred[pred_i + 1]):
                    eval = Eval([ent_gold], [ent_pred, sent._.ents_pred[pred_i + 1]], sent, True, "splitted")
                    eval_list.append(eval)
                    gold_i = gold_i + 1
                    pred_i = pred_i + 2
                elif is_overlapping_missing_ent(ent_gold, ent_pred):
                    # controlamos el caso "carrot cake vegan friendly" -> "carrot" "cake vegan friendly"
                    if ent_pred.start == ent_gold.start and ent_pred.end < ent_gold.end and pred_i + 1 < len(sent._.ents_pred) and gold_i + 1 != len(sent.ents) and sent._.ents_pred[pred_i + 1].start < ent_gold.end:
                        eval = Eval([ent_gold, sent.ents[gold_i + 1]], [ent_pred, sent._.ents_pred[pred_i + 1]], sent, True, "missegmented")
                        gold_i = gold_i + 2
                        pred_i = pred_i + 2
                    else:
                        eval = Eval([ent_gold], [ent_pred], sent, True, "overlap_missed")
                        gold_i = gold_i + 1
                        pred_i = pred_i + 1
                    eval_list.append(eval)
                elif is_overlapping_spurious_missing_ent(ent_gold, ent_pred):
                    # controlamos el caso "carrot cake vegan friendly" -> "cake vegan friendly"
                    if gold_i + 1 != len(sent.ents) and ent_pred.start > ent_gold.start and ent_pred.start < ent_gold.end and ent_pred.end == sent.ents[gold_i + 1].end :
                        eval1 = Eval([ent_gold], [], sent, True, "missed")
                        eval2 = Eval([sent.ents[gold_i + 1]], [ent_pred], sent, True, "missegmented")
                        gold_i = gold_i + 2
                        pred_i = pred_i + 1
                        eval_list.extend([eval1, eval2])
                    else:
                        eval = Eval([ent_gold], [ent_pred], sent, True, "overlap_spurious_missed")
                        eval_list.append(eval)
                        gold_i = gold_i + 1
                        pred_i = pred_i + 1
                elif gold_i + 1 != len(sent.ents) and is_missegment_ent(ent_gold, sent.ents[gold_i + 1], ent_pred):
                    # we now check if it was a perfect missegmentation (1 error only should be accounted) or not ("leggings animal" + "print" vs "leggings animal"; "carrot" + "cake vegan friendly" vs "cake vegan friendly")
                    if pred_i + 1 < len(sent._.ents_pred) and sent._.ents_pred[pred_i + 1].end == sent.ents[gold_i + 1].end:
                        eval = Eval([ent_gold, sent.ents[gold_i + 1]], [ent_pred], sent, True, "missegmented")
                        gold_i = gold_i + 2
                        pred_i = pred_i + 2
                    else:
                        eval = Eval([ent_gold], [ent_pred], sent, True, "missegmented")
                        gold_i = gold_i + 1
                        pred_i = pred_i + 1
                    eval_list.append(eval)

                elif is_overlapping_spurious_ent(ent_gold, ent_pred):
                    eval = Eval([ent_gold], [ent_pred], sent, True, "overlap_spurious")
                    eval_list.append(eval)
                    gold_i = gold_i + 1
                    pred_i = pred_i + 1

                elif ent_gold.start_char < ent_pred.start_char:
                    eval = Eval([ent_gold], [], sent, True, "missed")
                    eval_list.append(eval)
                    gold_i = gold_i + 1
                elif ent_gold.start_char > ent_pred.start_char:
                    eval = Eval([], [ent_pred], sent, True, "spurious")
                    eval_list.append(eval)
                    pred_i = pred_i + 1
                else:
                    print(ent_gold)
                    print(ent_pred)
            if pred_i < len(sent._.ents_pred): # predicted ents are left; they will be spurious
                for i in range(pred_i, len(sent._.ents_pred)):
                    spurious_ent = sent._.ents_pred[i]
                    eval = Eval([], [spurious_ent], sent, True, "spurious")
                    eval_list.append(eval)
            if gold_i < len(sent.ents): # gold ents are left; they will be missed
                for i in range(gold_i, len(sent.ents)):
                    missed_ent = sent.ents[i]
                    eval = Eval([missed_ent], [], sent, True, "missed")
                    eval_list.append(eval)
        self.evaluation = eval_list

    def set_extensions(self, split: Doc):
        for tok in split:
            tok._.is_perplexing = is_perplexing(tok)
            tok._.is_sentence_initial = tok.is_sent_start
            tok._.is_quotated = False if tok.is_sent_start or tok.is_sent_end else tok.doc[tok.i - 1] in QUOTATIONS or \
                                                                                   tok.doc[tok.i - 1] in QUOTATIONS
            tok._.is_upper = tok.text.isupper()
            tok._.is_titlecase = tok.text.istitle()

    def set_training_extensions(self, split: Doc):
        Span.set_extension("is_unseen", default=None, force=True)
        Span.set_extension("is_unseen_tagging", default=None, force=True)
        Span.set_extension("is_type_confusable", default=None, force=True)
        ## Extensions that depend on outside sources (training set, lexicon, gazetteer)
        Token.set_extension("is_unseen_token", default=None, force=True)
        Token.set_extension("is_unseen_tagging", default=None, force=True)
        Token.set_extension("is_type_confusable", default=None, force=True)
        Doc.set_extension("has_unseen_token", getter=has_unseen, force=True)
        Span.set_extension("has_unseen_token", getter=has_unseen, force=True)
        Doc.set_extension("has_unseen_tagging", getter=has_unseen_tagging, force=True)
        Span.set_extension("has_unseen_tagging", getter=has_unseen_tagging, force=True)
        Doc.set_extension("has_type_confusable", getter=has_type_confusable, force=True)
        Span.set_extension("has_type_confusable", getter=has_type_confusable, force=True)

        Doc.set_extension("has_unseen_ent", getter=has_unseen_ent, force=True)
        Span.set_extension("has_unseen_ent", getter=has_unseen_ent, force=True)
)

        for tok in split:
            tok._.is_unseen_token = tok.text not in self.previously_seen_tokens
            tok._.is_unseen_tagging = False if tok._.is_unseen_token else get_full_bio_tag(tok) not in \
                                                                          self.previously_seen_tokens[tok.text]
            tok._.is_type_confusable = False if tok._.is_unseen_token or not tok.text.isalpha() or tok.text in TYPE_CONFUSABLE_EXCLUDE else len(
                self.previously_seen_tokens[tok.text]) > 1 and len(tok.text) > 2

        for ent in split.ents:
            ent._.is_unseen = ent.text.lower() not in self.previously_seen_ents
            ent._.is_type_confusable = ent.text.lower() in self.previously_seen_ents and len(self.previously_seen_ents[ent.text.lower()]) > 1
            ent._.is_unseen_tagging = True if ent.text.lower() in self.previously_seen_ents and ent.label_ not in self.previously_seen_ents[ent.text.lower()] else False

    def update_outsourced_extensions(self, split: Doc):
        tokenized_gazetteer = {token for entity in self.gazetteer for token in entity.split()}  if  self.gazetteer else {}
        for tok in split:
            tok._.is_in_lexicon = tok.text.lower() in self.lexicon #if tok.is_sent_start else tok.text in self.lexicon
            tok._.is_in_gazetteer = tok.text.lower() in tokenized_gazetteer
            
        for ent in split.ents:
            ent._.is_in_gazetteer = ent.text in self.gazetteer
            


    def is_quotated(span):
        if span.doc[span.start].is_sent_start or span.doc[span.end - 1].is_sent_end:
            return False
        return span.doc[span.start - 1].text in QUOTATIONS and span.doc[span.end].text in QUOTATIONS

    def set_previously_seen_tokens(self) -> Dict[str, Set[str]]:
        previously_seen = defaultdict(set)
        for tok in self.training:
            previously_seen[tok.text].add(get_full_bio_tag(tok))
        return dict(previously_seen)

    def set_previously_seen_ents(self) -> Dict[str, Set[str]]:
        previously_seen_ents = defaultdict(set)
        for ent in self.training.ents:
            previously_seen_ents[ent.text.lower()].add(ent.label_)
        return dict(previously_seen_ents)

    @classmethod
    def from_conll_binary(cls, path_to_test_binary: str, **kwargs):
        test_doc: Doc = Cascabel._load_binary(path_to_test_binary)
        declare_extensions()
        training_file = kwargs.get('training', None)
        training_doc = Cascabel._load_binary(training_file) if training_file else None
        lexicon = kwargs.get('lexicon', None)
        gazetteer = kwargs.get('gazetteer', None)
        lexicon = Cascabel._load_lexicon(lexicon) if lexicon else set()
        gazetteer = Cascabel._load_lexicon(gazetteer) if gazetteer else set()
        prediction_file = kwargs.get('prediction', None)
        prediction_doc: Doc = Cascabel._load_binary(prediction_file) if prediction_file else None
        return cls(test_doc, training_doc, lexicon, gazetteer, prediction_doc)

    def spans_per_length(self, split: Doc) -> Counter:
        spans_per_length = defaultdict(set)
        for ent in split.ents:
            spans_per_length[len(ent)].add(ent)
        return spans_per_length

    def spans_per_length_distribution(self, split: Doc) -> Counter:
        spans_per_length = defaultdict(int)
        for ent in split.ents:
            spans_per_length[len(ent)] = spans_per_length[len(ent)] + 1
        return spans_per_length

    def span_per_sentence_distribution(self, split: Doc) -> Counter:
        spans_per_sentence = [len(sent.ents) for sent in split.sents]
        return Counter(spans_per_sentence)

    def spans_in_split(self, split: Doc) -> int:
        return len(list(split.ents))

    def span_length_distribution(self, split: Doc) -> Counter:
        span_lengths = [len(ent) for ent in split.ents]
        return Counter(span_lengths)

    def label_distribution(self, split: Doc) -> Counter:
        span_labels = [ent.label_ for ent in split.ents]
        return Counter(span_labels)


    @classmethod
    def _load_lexicon(self, path_to_lexicon: str) -> Set:
        return set(line.strip() for line in open(path_to_lexicon, encoding="utf-8"))


    @classmethod
    def _load_binary(self, path_to_binary: str) -> Doc:
        doc_bin = DocBin().from_disk(path_to_binary)
        return Doc.from_docs(list(doc_bin.get_docs(Vocab(strings=[]))))

    @classmethod
    def from_tokens(cls, tokens: List[str], tags: List[str]):
        declare_extensions()
        doc = Cascabel.tokens_to_doc(tokens, tags)
        return cls(doc)

    @classmethod
    def from_sentences(cls, sentences: List[Tuple[List[str], List[str]]]):
        declare_extensions()
        doc = Cascabel.sentences_to_doc(sentences)
        return cls(doc)

    @classmethod
    def tokens_to_doc(cls, tokens: List[str], tags: List[str]) -> Doc:
        is_head = [True] + [False] * (len(tokens) - 1)
        doc = Doc(Vocab(strings=[]), ents=tags, words=tokens, sent_starts=is_head)
        return doc

    def set_training_from_tokens(self, tokens: List[str], tags: List[str]):
        training_doc = Cascabel.tokens_to_doc(tokens, tags)
        self.training = training_doc
        self.previously_seen_tokens = self.set_previously_seen_tokens()
        self.previously_seen_ents = self.set_previously_seen_ents()
        self.set_training_extensions(self.test)

    def set_prediction_from_tokens(self, tokens: List[str], tags: List[str]):
        prediction_doc = Cascabel.tokens_to_doc(tokens, tags)
        self.prediction = prediction_doc
        self.add_prediction_layer()
        self.set_eval_instances()

    def set_prediction_from_sentences(self, sents: List[Tuple[List[str], List[str]]]):
        sents = fix_forbidden_transitions(sents)
        prediction_doc = Cascabel.sentences_to_doc(sents)
        self.prediction = prediction_doc
        self.add_prediction_layer()
        self.set_eval_instances()
        self.set_extensions(self.prediction)
        if self.training:
            self.set_training_extensions(self.prediction)

    @classmethod
    def from_conll_sentences(cls, test: List[Tuple[List[str], List[str]]], **kwargs):
        test_doc: Doc = Cascabel.sentences_to_doc(test)
        declare_extensions()
        training = kwargs.get('training', None)
        training_doc = Cascabel.sentences_to_doc(training) if training else None
        prediction_sentences = kwargs.get('prediction', None)
        prediction_doc: Doc = Cascabel.sentences_to_doc(prediction_sentences) if prediction_sentences else None
        return cls(test_doc, training_doc, None, None, prediction_doc)

    @classmethod
    def sentences_to_doc(cls, sentences: List[Tuple[List[str], List[str]]]) -> Doc:
        flattened_tokens = []
        flattened_tags = []
        is_head = []
        for sent, tags in sentences:
            for i, (token, tag) in enumerate(zip(sent, tags)):
                flattened_tokens.append(token)
                flattened_tags.append(tag)
                if i == 0:
                    is_head.append(True)
                else:
                    is_head.append(False)
        doc = Doc(Vocab(strings=[]), ents=flattened_tags, words=flattened_tokens, sent_starts=is_head)
        return doc



    def classify_instances(self) -> Dict:
        my_dict = defaultdict(list)
        for extension in self.test._.doc_extensions:
            my_dict[extension] = []
            for sent in self.test.sents:
                if getattr(sent._, extension):
                    my_dict[extension].append((sent, getattr(sent._, extension)))

        return dict(my_dict)

    def get_test_buckets(self) -> Dict:
        my_dict = dict()
        for extension in self.test._.doc_extensions:
            extension_dict = defaultdict(list)
            for sent in self.test.sents:
                if getattr(sent._, extension):
                    extension_dict["instances"].append((sent, getattr(sent._, extension)))
            extension_dict["sentences"] = len(extension_dict["instances"])
            extension_dict["percentage"] = float('%.2f' %(100*extension_dict["sentences"]/len(self.test.ents)))
            my_dict[extension] = dict(extension_dict)
        return dict(my_dict)


    def get_prediction_buckets(self) -> Dict:
        my_dict = dict()
        for extension in self.prediction._.doc_extensions:
            extension_dict = defaultdict(list)
            for ent in self.prediction.ents:
                if getattr(ent._, extension):
                    extension_dict["instances"].append((ent, getattr(ent._, extension)))
            extension_dict["sentences"] = len(extension_dict["instances"])
            extension_dict["percentage"] = float('%.2f' %(100*extension_dict["sentences"]/self.sent_len(self.test)))
            my_dict[extension] = dict(extension_dict)
        return dict(my_dict)

    def doc_to_lowercase(self):
        return Doc(Vocab(strings=[]), ents=[get_full_bio_tag(tok) for tok in self.test], words=[str(tok).lower() for tok in self.test], sent_starts=[tok.is_sent_start for tok in self.test])

    def doc_to_titlecase(self):
        return Doc(Vocab(strings=[]), ents=[get_full_bio_tag(tok) for tok in self.test], words=[str(tok).capitalize() for tok in self.test], sent_starts=[tok.is_sent_start for tok in self.test])

    def doc_to_uppercase(self):
        return Doc(Vocab(strings=[]), ents=[get_full_bio_tag(tok) for tok in self.test], words=[str(tok).upper() for tok in self.test], sent_starts=[tok.is_sent_start for tok in self.test])

    def ents_to_lowercase(self):
        return Doc(Vocab(strings=[]), ents=[get_full_bio_tag(tok) for tok in self.test], words=[str(tok).lower() if tok.ent_iob_ != "O" else str(tok) for tok in self.test], sent_starts=[tok.is_sent_start for tok in self.test])

    def ents_to_titlecase(self):
        return Doc(Vocab(strings=[]), ents=[get_full_bio_tag(tok) for tok in self.test], words=[str(tok).capitalize() if tok.ent_iob_ != "O" else str(tok) for tok in self.test], sent_starts=[tok.is_sent_start for tok in self.test])

    def ents_to_uppercase(self):
        return Doc(Vocab(strings=[]), ents=[get_full_bio_tag(tok) for tok in self.test], words=[str(tok).upper() if tok.ent_iob_ != "O" else str(tok) for tok in self.test], sent_starts=[tok.is_sent_start for tok in self.test])

    def add_quotations_to_spans(self, doc_to_lowercase = False, doc_to_titlecase = False, doc_to_uppercase = False, span_to_lowercase = False, span_to_titlecase = False, span_to_uppercase = False):
        span_to_lowercase = True if doc_to_lowercase else span_to_lowercase
        span_to_titlecase = True if doc_to_titlecase else span_to_titlecase
        span_to_uppercase = True if doc_to_uppercase else span_to_uppercase
        new_tokens = []
        new_tags = []
        new_heads = []

        lowestbound = 0
        for ent in self.test.ents:
            for tok in self.test[lowestbound:ent.start]:
                tok_cased = apply_casing(tok, doc_to_lowercase, doc_to_titlecase, doc_to_uppercase)
                new_tokens.append(tok_cased)
                new_tags.append("O")
                new_heads.append(tok.is_sent_start)
            if not ent._.has_quotations:
                new_tokens.append("\"")
                new_tags.append("O")
                new_heads.append(ent[0].is_sent_start)
            for tok in ent:
                tok_cased = apply_casing(tok, span_to_lowercase, span_to_titlecase, span_to_uppercase)
                new_tokens.append(tok_cased)
                new_tags.append(get_full_bio_tag(tok))
                new_heads.append(False)
            if not ent._.has_quotations:
                new_tokens.append("\"")
                new_tags.append("O")
                new_heads.append(False)
            lowestbound = ent.end
        for tok in self.test[lowestbound:] :
            tok_cased = apply_casing(tok, doc_to_lowercase, doc_to_titlecase, doc_to_uppercase)
            new_tokens.append(tok_cased)
            new_tags.append(get_full_bio_tag(tok))
            new_heads.append(tok.is_sent_start)

        return Doc(Vocab(strings=[]), ents=new_tags, words=new_tokens, sent_starts=new_heads)

    def remove_quotations_from_spans(self, doc_to_lowercase = False, doc_to_titlecase = False, doc_to_uppercase = False, span_to_lowercase = False, span_to_titlecase = False, span_to_uppercase = False):
        span_to_lowercase = True if doc_to_lowercase else span_to_lowercase
        span_to_titlecase = True if doc_to_titlecase else span_to_titlecase
        span_to_uppercase = True if doc_to_uppercase else span_to_uppercase
        new_tokens = []
        new_tags = []
        new_heads = []

        lowestbound = 0
        for ent in self.test.ents:
            for tok in self.test[lowestbound:ent.start]:
                tok_cased = apply_casing(tok, doc_to_lowercase, doc_to_titlecase, doc_to_uppercase)
                new_tokens.append(tok_cased)
                new_tags.append("O")
                new_heads.append(tok.is_sent_start)
            if ent._.has_quotations:
                new_tokens.pop()
                new_tags.pop()
                was_sentence_initial = new_heads.pop()
            for i, tok in enumerate(ent):
                tok_cased = apply_casing(tok, span_to_lowercase, span_to_titlecase, span_to_uppercase)
                new_tokens.append(tok_cased)
                new_tags.append(get_full_bio_tag(tok))
                if i == 0 and ent._.has_quotations:
                    new_heads.append(was_sentence_initial)
                else:
                    new_heads.append(tok.is_sent_start)
            if ent._.has_quotations:
                lowestbound = ent.end + 1
            else:
                lowestbound = ent.end
        for tok in self.test[lowestbound:]:
            tok_cased = apply_casing(tok, doc_to_lowercase, doc_to_titlecase, doc_to_uppercase)
            new_tokens.append(tok_cased)
            new_tags.append(get_full_bio_tag(tok))
            new_heads.append(tok.is_sent_start)

        return Doc(Vocab(strings=[]), ents=new_tags, words=new_tokens, sent_starts=new_heads)

    def get_titlecase_spans(self, doc: Doc):
        return [ent for ent in doc.ents if not ent._.is_titlecase]

    def get_multitoken_spans(self, doc: Doc):
        return [ent for ent in doc.ents if ent._.is_multitoken]

    def get_singletoken_spans(self, doc: Doc):
        return [ent for ent in doc.ents if not ent._.is_multitoken]

    def get_sent_init_spans(self, doc: Doc):
        return [ent for ent in doc.ents if ent._.is_sentence_initial]



    def get_upper_spans(self, doc):
        return [ent for ent in doc.ents if  ent._.is_upper]

    def get_quoted_spans(self, doc):
        return [ent for ent in doc.ents if ent._.is_quotated]

    def get_adjacent_spans(self, doc):
        return [ent for ent in doc.ents if ent._.is_collocated]


    def get_training_dimensions(self):
        self.get_dimensions(self.training, is_test=False)

    def get_test_dimensions(self):
        self.get_dimensions(self.test)

    def get_unseen_spans(self, doc: Doc):
        return [ent for ent in doc.ents if ent._.is_unseen_ent]

    def get_unseen_tag_spans(self, doc: Doc):
        return [ent for ent in doc.ents if ent._.has_unseen_tagging]

    def get_type_conf_spans(self, doc: Doc):
        return [ent for ent in doc.ents if ent._.has_type_confusable]

    def get_dimensions(self, doc: Doc, is_test=True) -> Dict:
        mydict = dict()
        mydict["multitoken"] = round(100*len(self.get_multitoken_spans(doc))/len(doc.ents),2)
        mydict["singletoken"] = round(100*len(self.get_singletoken_spans(doc))/len(doc.ents),2)
        mydict["sent_initial"] = round(100*len(self.get_sent_init_spans(doc))/len(doc.ents),2)
        mydict["titlecase"] = round(100*len(self.get_titlecase_spans(doc)) / len(doc.ents),2)
        mydict["uppercase"] = round(100*len(self.get_upper_spans(doc)) / len(doc.ents),2)
        mydict["quoted"] = round(100*len(self.get_quoted_spans(doc)) / len(doc.ents),2)
        mydict["adjacent"] = round(100*len(self.get_adjacent_spans(doc)) / len(doc.ents),2)

        if is_test and self.training:
            mydict["unseen"] = round(100*len(self.get_unseen_spans(doc)) / len(doc.ents),2)
            mydict["unseen_tag"] = round(100*len(self.get_unseen_tag_spans(doc)) / len(doc.ents),2)
            mydict["type_conf"] = round(100*len(self.get_type_conf_spans(doc)) / len(doc.ents),2)

        return mydict








