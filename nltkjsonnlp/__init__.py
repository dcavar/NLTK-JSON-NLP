#!/usr/bin/env python3

"""
(C) 2019 Damir Cavar, Oren Baldinger, Maanvitha Gongalla, Anurag Kumar, Murali Kammili, Boli Fang

Wrappers for NLTK to JSON-NLP output format.

Licensed under the Apache License 2.0, see the file LICENSE for more details.

Brought to you by the NLP-Lab.org (https://nlp-lab.org/)!
"""


name = "nltkjsonnlp"

__version__ = "0.0.1"

import json, logging, functools
import jsonnlp 
from collections import OrderedDict, Counter, defaultdict
import functools
import os
from os.path import join
import nltk
from typing import List, Tuple
from nltk import __version__, PunktSentenceTokenizer, TreebankWordTokenizer, PorterStemmer, tagset_mapping, ne_chunk, pos_tag
from nltk.chunk import tree2conlltags
from nltk.corpus import wordnet
from nltk.corpus import verbnet as vn
from nltk.parse import malt
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize.api import TokenizerI
from langdetect import detect
from jsonnlp import base_nlp_json, base_document
import pycountry



def cache_it(func):
    """A decorator to cache function response based on params. Add it to top of function as @cache_it."""

    global __cache

    @functools.wraps(func)
    def cached(*args):
        f_name = func.__name__
        s = ''.join(map(str, args))
        if s not in __cache[f_name]:
            __cache[f_name][s] = func(*args)
        return __cache[f_name][s]
    return cached


#@cache_it
def get_wordnet_pos(penn_tag: str) -> str:
    """Returns the Penn tag as a WordNet tag."""

    if penn_tag.startswith('J'):
        return wordnet.ADJ
    elif penn_tag.startswith('V'):
        return wordnet.VERB
    elif penn_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


@cache_it
def get_sentence_tokenizer():
    """Returns the sentence tokenizer. This function cashes the sentencer object."""

    return PunktSentenceTokenizer()


@cache_it
# def get_tokenizer(model: str) -> TokenizerI:
def get_tokenizer(params: dict):
    """Returns the Treebank Word Tokenizer."""

    model = params.get('tokenizer', '').lower()
    if model == 'punkt':
        return WordPunctTokenizer()
    if model != '' and model != 'treebank':
        raise ModuleNotFoundError(f'No such tokenizer {model}!')
    return TreebankWordTokenizer()


@cache_it
def get_lemmatizer():
    """Return the WordNet Lemmatizer. This function cashes the lemmatizer object."""

    return WordNetLemmatizer().lemmatize


@cache_it
def get_stemmer():
    """ """
    # https://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization
    return PorterStemmer().stem


@cache_it
def get_tag_mapper(lang: str) -> dict:
    """ """

    if lang == 'en':
        return tagset_mapping('en-ptb', 'universal')
    return {}


@cache_it
def get_parser():
    """Returns the MaltParser object.  This function cashes the object."""

    try:
        malt_path = os.environ['MALT_PATH']
        model_path = os.environ['MODEL_PATH']
    except KeyError:
        return None
    return malt.MaltParser(malt_path, model_path)



def process(text: str, params: dict) -> OrderedDict:
    """This function is the main process function that processes the entire text."""

    # set JSON-NLP
    j: OrderedDict = base_document()
    t: OrderedDict = base_nlp_json()
    t['DC.source'] = 'NLTK {}'.format(__version__)
    t['documents'].append(j)
    j['text'] = text

    # collect parsers
    lemmatizer = get_lemmatizer()
    tokenizer = get_tokenizer(params)
    sentence_tokenizer = get_sentence_tokenizer()
    stemmer = get_stemmer()
    parser = get_parser()
    language = Counter()

    # tokenize and tag
    tokens: List[str] = tokenizer.tokenize(text)
    tokens_tagged: List[tuple] = nltk.pos_tag(tokens)
    conll_tagged = tree2conlltags(ne_chunk(tokens_tagged))

    offset_list: List[Tuple[int, int]] = list(tokenizer.span_tokenize(text))

    token_list: List[dict] = []
    for token_idx, token_tuple in enumerate(tokens_tagged):
        token = token_tuple[0]
        pos_tag = token_tuple[1]
        wordnet_pos = get_wordnet_pos(pos_tag)
        entity_tag = conll_tagged[token_idx][2].split("-")

        if wordnet_pos != '':
            synsets = wordnet.synsets(token, pos=wordnet_pos)
        else:
            synsets = wordnet.synsets(token)
        sys_id = 0
        sys_list = []
        for syn in synsets:
            s_hypo = set([x.lemma_names()[0] for x in syn.hyponyms()])
            s_hyper = set([x.lemma_names()[0] for x in syn.hypernyms()])
            s_examples = [x for x in syn.examples()]

            s = {
                'wordnet_id': syn.name(),
                'id': sys_id,
                'synonym': syn.lemma_names()[1:],
                'hyponym': list(s_hypo),
                'hypernym': list(s_hyper),
                'examples': s_examples,
                'definition': syn.definition()
            }

            if len(s['synonym']) == 0: s.pop('synonym')
            if len(s['hyponym']) == 0: s.pop('hyponym')
            if len(s['hypernym']) == 0: s.pop('hypernym')
            if len(s['examples']) == 0: s.pop('examples')
            if len(s['definition']) == 0: s.pop('definition')

            if s:
                sys_list.append(s)
            sys_id += 1

        verb_list = []
        vn_classids = vn.classids(token)
        for classid in vn_classids:
            verb_list.append({'class_id': classid,
                              'frames': vn.frames(classid)})

        t = {
            'id': token_idx,
            'text': token,
            'lemma': lemmatizer(token, wordnet_pos) if wordnet_pos else lemmatizer(token),
            'stem': stemmer(token),
            'pos': pos_tag,
            'entity': entity_tag[1] if len(entity_tag) > 1 else "",
            'entity_iob': entity_tag[0],
            'overt': True,
            'characterOffsetBegin': offset_list[token_idx][0],
            'characterOffsetEnd': offset_list[token_idx][1],
            'synsets': sys_list,
            'verbnet': verb_list
        }
        if len(t['synsets']) == 0: t.pop('synsets')
        if len(t['verbnet']) == 0: t.pop('verbnet')
        token_list.append(t)

    j['tokenList'] = token_list

    # sentence and dependency parsing
    sent_list = []
    token_from = 0
    sentence_tokens = sentence_tokenizer.sentences_from_tokens(tokens)
    sentence_texts = sentence_tokenizer.sentences_from_text(text)

    # check whether MALT parser is loaded! DC
    if parser:
        for sent_idx, sent in enumerate(zip(sentence_tokens, sentence_texts)):
            # Detecting language of each sentence
            la = pycountry.languages.get(alpha_2=detect(sent[1]))  
            token_to = token_from + len(sent[0]) - 1
            dg = parser.parse_one(sent[1].split())
            s = {
                'id': sent_idx,
                'text': sent[1],
                'tokenFrom': token_from,
                'tokenTo': token_to,
                'tokens': list(range(token_from, token_to))
            }

            for token in dg.nodes:
                head = dg.nodes[token]['head']
                head_word = [dg.nodes[i]['word'] for i in dg.nodes if dg.nodes[i]['address'] == head]
                if len(head_word) > 0:
                    j['dependenciesBasic'].append({
                        'governor': head_word[0],
                        'dependent': dg.nodes[token]['word'],
                        'type': dg.nodes[token]['rel']
                    })
                else:
                    j['dependenciesBasic'].append({
                        'governor': 'null',
                        'dependent': dg.nodes[token]['word'],
                        'type': dg.nodes[token]['rel']
                    })
                if j['dependenciesBasic'][-1]['governor'] == 'null' or j['dependenciesBasic'][-1]['dependent'] == 'null' \
                        or j['dependenciesBasic'][-1]['type'] == 'null':
                    j['dependenciesBasic'].pop()
            token_from = token_to
            language[la.name] += 1
            sent_list.append(s)
        j['sentences'] = sent_list

    if params['language']:
        t['DC.language'] = params['language']
    else:
        # only if language has some elements can we check for max!!! DC
        if len(token_list) > 4 and language:
            t['DC.language'] = max(language)
        else:
            t['DC.language'] = ''

    # TODO:
    # 1. Schema: clauses, coreferences, constituents, expressions, paragraphs
    # 2. fields: token: sentiment, embeddings; sentence: sentiment, complex, type, embeddings

    return j
