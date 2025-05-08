from spacy.tokens import Doc, Span, Token
from cascabel.constants import *
import pylabeador
from pylabeador import *
from typing import List, Dict, Set, Tuple


def check_type(ent):
    if ent._.has_adjacent_entities:
        return "adjacent"
    elif is_ambiguous(ent):
        return "ambiguous"
    elif is_mixed_ambiguous(ent):
        return "mixed ambiguous"
    elif ent._.is_perplexing_ent:
        return "non-compliant"
    elif ent._.is_not_perplexing_ent:
        return "compliant"
    elif ent._.is_mixed_perplexing_ent:
        return "mixed compliant"
    else:
        print(ent)

def check_length(ent):
    if ent._.is_multitoken:
        return "multi"
    return "single"

def check_position(ent):
    if ent._.is_sentence_initial:
        return "ini"
    return "mid"

def check_quotes(ent):
    if ent._.has_quotations:
        return "quoted"
    return "unquoted"

def check_casing(ent):
    if ent._.is_titlecase:
        return "titlecase"
    if ent._.is_upper:
        return "uppercase"
    else:
        return "lowercase"

def apply_casing(tok: Token, to_lowercase: bool, to_titlecase: bool, to_uppercase: bool) -> str:
    if to_lowercase:
        return str(tok).lower()
    elif to_titlecase:
        return str(tok).capitalize()
    elif to_uppercase:
        return str(tok).upper()
    else:
        return str(tok)

def p_r_f1(tp: int, fp: int, fn: int):
    precision = tp/(fp+tp) if (fp+tp) > 0 else 0
    recall = tp/(fn+tp) if (fn+tp) > 0 else 0
    f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    return precision, recall, f1


def get_p_r_f(fp, fn, tp):
    return tp / (fp + tp), tp / (fn + tp), (2 * tp / (2 * tp + fp + fn))

def sent_len(split: Doc) -> int:
    return len(list(split.sents))


def token_len(split: Doc) -> int:
    return len(list(split))

def fix_forbidden_transitions(sents: List[Tuple[List[str], List[str]]]) -> List[Tuple[List[str], List[str]]]:
    new_sents = []
    for sent, tagging in sents:
        previous_tag = None
        new_tagging = []
        for tag in tagging:
            label = tag.split("-")[1] if len(tag.split("-")) > 1 else None
            if tag.startswith("I"):
                if previous_tag is None or previous_tag == "O":
                    tag = "B-" + label
            new_tagging.append(tag)
            previous_tag = tag
        new_sents.append((sent,new_tagging))
    return new_sents



def get_full_bio_tag(token: Token):
    full_tag = token.ent_iob_ if token.ent_iob_ == "O" else token.ent_iob_ + "-" + token.ent_type_
    return full_tag

def mark_token_error(new_ent: Span):
    for tok in new_ent:
        tok._.is_correct = False
        tok._.error_type = new_ent._.error_type

def mark_token_correct(new_ent: Span):
    for tok in new_ent:
        tok._.is_correct = True

def mark_token_error_overlap(gold_ent: Span, pred_ent: Span):
    for i in range(gold_ent.start, gold_ent.end):
        pred_tok = pred_ent.doc[i]
        if pred_tok.ent_iob_ == "O":
            pred_tok._.is_correct = False
            pred_tok._.error_type = "missed"
        else:
            pred_tok._.is_correct = True
    for i in range(pred_ent.start, pred_ent.end):
        gold_tok = gold_ent.doc[i]
        if gold_tok.ent_iob_ == "O":
            pred_ent.doc[i]._.is_correct = False
            pred_ent.doc[i]._.error_type = "spurious"
        else:
            pred_ent.doc[i]._.is_correct = True


def is_fused_ent(ent1: Span, ent2: Span, ent_pred: Span) -> bool:
    return ent_pred.start == ent1.start and ent_pred.end >= ent2.end

def is_missegment_ent(ent1: Span, ent2: Span, ent_pred: Span) -> bool:
    return ent_pred.start >= ent1.start and ent_pred.start < ent1.end and ent_pred.end >= ent2.start

def is_splitted_ent(ent_gold: Span, ent_pred1: Span, ent_pred2: Span) -> bool:
    return ent_gold.start <= ent_pred1.start and ent_gold.end >= ent_pred2.end

def is_same_ent(ent_gold: Span, ent_pred: Span) -> bool:
    return ent_gold.start == ent_pred.start and  ent_gold.end == ent_pred.end and  ent_gold.label_ == ent_pred.label_


def is_overlapping_missing_ent(ent_gold: Span, ent_pred: Span) -> bool:
    if ent_pred.start >= ent_gold.start and ent_pred.end <= ent_gold.end:
        return True

def is_overlapping_spurious_ent(ent_gold: Span, ent_pred: Span) -> bool:
    if ent_pred.start <= ent_gold.start and ent_pred.end >= ent_gold.end:
        return True

def is_overlapping_spurious_missing_ent(ent_gold: Span, ent_pred: Span) -> bool:
    if (ent_pred.start < ent_gold.start and ent_pred.end < ent_gold.end and ent_pred.end> ent_gold.start) or (ent_pred.start > ent_gold.start and ent_pred.start < ent_gold.end and ent_pred.end > ent_gold.end):
        return True



def is_overlapping_ent(ent_gold: Span, ent_pred: Span) -> bool:
     if ent_pred.start > ent_gold.start and ent_pred.start < ent_gold.end:
         return True
     if ent_pred.end > ent_gold.start and ent_pred.end < ent_gold.end:
         return True
     if ent_pred.start <= ent_gold.start and ent_pred.end >= ent_gold.end:
         return True
     return False

def is_type_swap(ent_gold: Span, ent_pred: Span) -> bool:
    return ent_gold.start == ent_pred.start and  ent_gold.end == ent_pred.end and  ent_gold.label_ != ent_pred.label_

# Define getter function
def is_multitoken(span):
    return len(span) > 1

# Define getter function
def is_sentence_initial(span):
    return span.doc[span.start].is_sent_start or  is_sentence_initicial_punctuation(span) # Quotation at the beginning of the sentence considered

def is_sentence_initicial_punctuation(span):
    """
    checks whether a span that is not considered sentence initial by spacy is indeed sentence initial because is exclusively preceeded by puntuation symbols
    """
    for i in range(span.sent.start, span.start):
        tok = span.doc[i]
        if not (tok.is_punct or is_left_punct(tok)):
            return False
    return True

def is_left_punct(tok: Token):
    return  tok.is_left_punct or tok.text in ["¿", "¡", "\"", "\'"]

def is_quotated(span):
    if span.doc[span.start].is_sent_start or span.doc[span.end - 1].is_sent_end:
        return False
    return span.doc[span.start - 1].text in QUOTATIONS and span.doc[span.end].text in QUOTATIONS


def is_upper(span):
    return span.text.isupper()


def is_titlecase(span):
    return span.text[0].isupper() and not is_sentence_initial(span)


def is_preceeded(span):
    if span.doc[span.start].is_sent_start:
        return False
    return span.doc[span.start - 1].ent_iob_ != "O"

def is_followed(span):
    if span.doc[span.end].is_sent_end:
        return False
    return span.doc[span.end].ent_iob_ != "O"

def has_multitoken(tokens): # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    return [ent for ent in tokens.ents if ent._.is_multitoken]

def has_sentence_initial(tokens): # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    return [ent for ent in tokens.ents if ent._.is_sentence_initial]


def has_quotations(tokens): # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    return [ent for ent in tokens.ents if ent._.is_quotated]

def has_no_quotations(tokens): # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    return [ent for ent in tokens.ents if not ent._.is_quotated]

def has_upper(tokens): # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    return [ent for ent in tokens.ents if ent._.is_upper]

def has_titlecase(tokens): # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    return [ent for ent in tokens.ents if ent._.is_titlecase]

def is_seen(span):
    return not span._.is_unseen

def is_unseen_span(span):
    return span._.is_unseen

def is_unseen_tagging(span):
    return span._.is_unseen_tagging

def is_type_confusable(span):
    return span._.is_type_confusable



def has_unseen(tokens):  # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    return [tok for tok in tokens if tok._.is_unseen_token]

def has_unseen_tagging(
        tokens):  # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    return [tok for tok in tokens if tok._.is_unseen_tagging]

def has_type_confusable(
        tokens):  # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    return [tok for tok in tokens if tok._.is_type_confusable]

def has_gazetteer_ent(
        tokens):  # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    """
    Returns the list of tokens found in the inclusion lexicon
    The inclusion lexicon should be a list of candidate tokens/gazetteer
    :param tokens:
    :return:
    """
    return [ent for ent in tokens.ents if ent._.is_in_gazetteer]

def has_unseen_ent(tokens):
    """
    Returns the list of ents that were not seen in training
    :param tokens: Doc/Span
    :return: List of Spans
    """
    return [ent for ent in tokens.ents if ent._.is_unseen_ent]

def has_gazetteer_O_token(
        tokens):  # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    """
    Returns the list of tokens found in a given gazetteer (a list of previously compiled entities) that are O
    The inclusion lexicon should be a list of candidate tokens/gazetteer
    :param tokens:
    :return:
    """
    return [tok for tok in tokens if tok._.is_in_gazetteer and tok.ent_iob_ == "O"]


def has_unregistered_token(
        tokens):  # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    """
    Return list of tokens that do not appear in the exclusion lexicon.
    The exclusion lexicon should be a list of expected/unremarkable words, ie the lexicon of a language
    This method returns the tokens in the sentence that were not found in that lexicon
    This method should return unsual words (typos, neologisms, nonce words) that yet labeled as O
    :param tokens:
    :return:
    """
    return [tok for tok in tokens if tok.text.isalpha() and not tok._.is_in_lexicon]

def has_unregistered_O_token(
        tokens):  # we dont specify type annotation for tokens so we can pass this method to Doc or Sentences (Span)
    """
    Return list of tokens labeled as O that do not appear in the exclusion lexicon.
    The exclusion lexicon should be a list of expected/unremarkable words, ie the lexicon of a language
    This method returns the tokens in the sentence that were not found in that lexicon
    This method should return unsual words (typos, neologisms, nonce words) that yet labeled as O
    :param tokens:
    :return:
    """
    return [tok for tok in tokens if tok.text.isalpha() and tok.ent_iob_ == "O" and not tok._.is_in_lexicon]

def has_adjacent_entities(tokens):
    """
    The sentence has an entity that is adjacent to another entity (such as "look total black").
    We look for a token labeled with a B tag whose preceeding token has a tag that is not O
    :param tokens:
    :return:
    """
    return [ent for ent in tokens.ents if ent._.is_collocated]

def has_error(tokens):
    """
    The sentence has an entity that is adjacent to another entity (such as "look total black").
    We look for a token labeled with a B tag whose preceeding token has a tag that is not O
    :param tokens:
    :return:
    """
    return [(ent,ent._.error_type) for ent in tokens.ents if ent._.is_correct == False]

def has_correct(tokens):
    """
    The sentence has an entity that is adjacent to another entity (such as "look total black").
    We look for a token labeled with a B tag whose preceeding token has a tag that is not O
    :param tokens:
    :return:
    """
    return [ent for ent in tokens.ents if ent._.is_correct]

def is_invalid_onset(onset: str) -> bool:
    if len(onset) > 1:
        return onset not in VALID_ONSETS
    return False

def is_invalid_coda(coda: str) -> bool:
    if len(coda) > 1:
        return coda not in VALID_CODAS
    return False

def is_invalid_ending(word: str) -> bool:
    if len(word) == 1:
        return False
    if word[-1] in VOWELS:
        return False
    return word[-2] not in VOWELS or word[-1] not in VALID_CONSONANT_ENDING

def is_invalid_transition(char1, char2):
    if (char1 == "b" and char2 != "b") or (char1 == "x" and char2 != "x"):
        return False
    if char1 in VALID_CONSONANT_ENDING and char1 != char2:
        return False
    if char1+char2 not in VALID_TRANSITIONS:
        return True
    return False

def contains_forbidden_pattern(word: str):
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in word:
            return True
    return False

def is_perplexing(token: Token):
    if not token.text.isalpha():
        return False
    if contains_forbidden_pattern(token.text.lower()):
        return True
    try:
        syllabified_word = syllabify_with_details(token.text.lower())
    except pylabeador.errors.HyphenatorError:
        return True
    for syl in syllabified_word.syllables:
        if is_invalid_onset(syl.onset):
            return True
        if is_invalid_coda(syl.coda):
            return True
        if is_invalid_ending(token.text.lower()):
            return True
    for first, second in zip(syllabified_word.syllables, syllabified_word.syllables[1:]):
        if first.coda and second.onset:
            if is_invalid_transition(first.coda[-1],second.onset[0]):
                return True
    return False

def has_perplexing(tokens):
    return tokens._.is_perplexing_ent or tokens._.is_mixed_perplexing_ent

def is_perplexing_ent(tokens): # know how
    return all([tok._.is_perplexing for tok in tokens if tok.text.isalpha()])

def is_not_perplexing_ent(tokens): # prime time
    return not [tok for tok in tokens if tok.text.isalpha() and tok._.is_perplexing]

def is_mixed_perplexing_ent(tokens): # total white
    return any([tok._.is_perplexing for tok in tokens if tok.text.isalpha()]) and any([not tok._.is_perplexing for tok in tokens if tok.text.isalpha()])

def is_ambiguous(tokens): # know how
    return all([tok._.is_in_lexicon for tok in tokens if tok.text.isalpha()])
    
def is_mixed_ambiguous(tokens): # total white
    return any([tok._.is_in_lexicon for tok in tokens if tok.text.isalpha()]) and any([not tok._.is_in_lexicon for tok in tokens if tok.text.isalpha()])

def has_in_lexicon(tokens):
    return [tok for tok in tokens if tok._.is_in_lexicon]


def set_lexicon_extension():
    Span.set_extension("is_in_gazetteer", default=None, force=True)
    Token.set_extension("is_in_gazetteer", default=None, force=True)
    Doc.set_extension("has_gazetteer_ent", getter=has_gazetteer_ent, force=True)
    Span.set_extension("has_gazetteer_ent", getter=has_gazetteer_ent, force=True)
    Token.set_extension("is_in_lexicon", default=None, force=True)
    Span.set_extension("has_in_lexicon", getter=has_in_lexicon, force=True)
    Span.set_extension("is_ambiguous", getter=is_ambiguous, force=True)
    Span.set_extension("is_mixed_ambiguous", getter=is_mixed_ambiguous, force=True)
    Doc.set_extension("has_unregistered_token", getter=has_unregistered_token, force=True)
    Span.set_extension("has_unregistered_token", getter=has_unregistered_token, force=True)





def declare_extensions():
    """
    set_extension must live here as an utils functions, not as a class method for Cascabel, bc Cascabel object is not created yet when this method is called
    :param self:
    :return:
    """
    Span.set_extension("is_multitoken", getter=is_multitoken, force=True)
    Span.set_extension("is_sentence_initial", getter=is_sentence_initial, force=True)
    Span.set_extension("is_quotated", getter=is_quotated, force=True)
    Span.set_extension("is_upper", getter=is_upper, force=True)
    Span.set_extension("is_titlecase", getter=is_titlecase, force=True)
    Span.set_extension("is_collocated", getter=is_preceeded, force=True)

    Doc.set_extension("has_multitoken", getter=has_multitoken, force=True)
    Span.set_extension("has_multitoken", getter=has_multitoken, force=True)

    Doc.set_extension("has_sentence_initial", getter=has_sentence_initial, force=True)
    Span.set_extension("has_sentence_initial", getter=has_sentence_initial, force=True)

    Doc.set_extension("has_quotations", getter=has_quotations, force=True)
    Span.set_extension("has_quotations", getter=has_quotations, force=True)

    Doc.set_extension("has_upper", getter=has_upper, force=True)
    Span.set_extension("has_upper", getter=has_upper, force=True)

    Doc.set_extension("has_titlecase", getter=has_titlecase, force=True)
    Span.set_extension("has_titlecase", getter=has_titlecase, force=True)



    Token.set_extension("is_sentence_initial", default=None, force=True)
    Token.set_extension("is_quotated", default=None, force=True)
    Token.set_extension("is_upper", default=None, force=True)
    Token.set_extension("is_titlecase", default=None, force=True)
    Token.set_extension("is_perplexing", default=None, force=True)

    Doc.set_extension("has_perplexing", getter=has_perplexing, force=True)
    Span.set_extension("has_perplexing", getter=has_perplexing, force=True)

    Doc.set_extension("is_perplexing_ent", getter=is_perplexing_ent, force=True)
    Span.set_extension("is_perplexing_ent", getter=is_perplexing_ent, force=True)

    Doc.set_extension("is_not_perplexing_ent", getter=is_not_perplexing_ent, force=True)
    Span.set_extension("is_not_perplexing_ent", getter=is_not_perplexing_ent, force=True)

    Doc.set_extension("is_mixed_perplexing_ent", getter=is_mixed_perplexing_ent, force=True)
    Span.set_extension("is_mixed_perplexing_ent", getter=is_mixed_perplexing_ent, force=True)


    Doc.set_extension("has_adjacent_entities", getter=has_adjacent_entities, force=True)
    Span.set_extension("has_adjacent_entities", getter=has_adjacent_entities, force=True)


    Span.set_extension("has_no_quotations", getter=has_no_quotations, force=True)
    Doc.set_extension("has_no_quotations", getter=has_no_quotations, force=True)





