#!/usr/bin/env python3

"""
(C) 2019 Damir Cavar, Oren Baldinger, Maanvitha Gongalla, Anurag Kumar, Murali Kammili, Boli Fang

Wrappers for NLTK to JSON-NLP output format.

Licensed under the Apache License 2.0, see the file LICENSE for more details.

Brought to you by the NLP-Lab.org (https://nlp-lab.org/)!
"""
from collections import OrderedDict

from nltk import __version__ as nltk_version, PorterStemmer, pos_tag
from nltk.corpus import framenet as fn
from nltk.corpus import verbnet as vn
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from pyjsonnlp import get_base, get_base_document, remove_empty_fields
from pyjsonnlp.pipeline import Pipeline
from pyjsonnlp.tokenization import segment

__version__ = "0.0.3"
name = "nltkjsonnlp"
__cache = {}


def get_wordnet_pos(penn_tag: str) -> str:
    # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
    if penn_tag.startswith('J'):
        return wordnet.ADJ
    elif penn_tag.startswith('V'):
        return wordnet.VERB
    elif penn_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def get_lemmatizer():
    # https://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization
    return WordNetLemmatizer().lemmatize


def get_stemmer():
    # https://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization
    return PorterStemmer().stem


def invoke_frame(token: str):
    word = token.lower()
    lu_list = [(i.name, i.definition) for i in fn.lus()]
    lu_temp = set([i for i in lu_list if word == i[0].split('.')[0]])
    frames = []

    for lu, def_ in lu_temp:
        fr = fn.frames_by_lemma(r'(?i)' + lu)
        # print(len(fr), fr[0].ID)
        if len(frames) == 0:
            frames.append(fr[0])
        else:
            if fr[0] not in frames:
                frames.append(fr[0])

    return frames


class NltkPipeline(Pipeline):
    @staticmethod
    def process(text='', lang='en', coreferences=False, constituents=False, dependencies=False, expressions=False,
                **kwargs) -> OrderedDict:
        # build nlp-json
        j: OrderedDict = get_base()
        j['meta']['DC.language'] = lang
        d: OrderedDict = get_base_document(1)
        j['documents'][d['id']] = d
        d['meta']['DC.source'] = 'NLTK {}'.format(nltk_version)
        j['meta']['DC.language'] = lang
        d['text'] = text

        # collect parsers
        lemmatizer = get_lemmatizer()
        stemmer = get_stemmer()

        # tokenization and pos
        words = []
        for sent in segment(text):
            for token in sent:
                words.append(token.value)

        # create the token list
        t_id = 1
        for word, xpos in pos_tag(words):
            wordnet_pos = get_wordnet_pos(xpos)
            lemma = lemmatizer(word, pos=wordnet_pos)

            # start the token
            t = {
                'id': t_id,
                'text': word,
                'stem': stemmer(word)
            }
            d['tokenList'][t['id']] = t
            t_id += 1

            # wordnet
            try:
                synsets = wordnet.synsets(lemma, pos=wordnet_pos)
                senses = {}
                for s in synsets:
                    hyponyms = [y for x in s.hyponyms() for y in x.lemma_names()]
                    hypernyms = [y for x in s.hypernyms() for y in x.lemma_names()]
                    synonyms = s.lemma_names()[1:]
                    examples = s.examples()
                    sense = {
                        'wordnetId': s.name(),
                        'definition': s.definition()
                    }
                    if synonyms:
                        sense['synonyms'] = synonyms
                    if hypernyms:
                        sense['hypernyms'] = hypernyms
                    if hyponyms:
                        sense['hyponyms'] = hyponyms
                    if examples:
                        sense['examples'] = examples

                    antonyms = []
                    for l in s.lemmas():
                        if l.antonyms():
                            for a in l.antonyms():
                                antonyms.append(a.name())
                    if antonyms:
                        sense['antonyms'] = antonyms

                    senses[sense['wordnetId']] = sense

                if senses:
                    t['synsets'] = senses
            except:
                pass

            # verbnet
            try:
                verbs = dict((class_id, {'classId': class_id, 'frames': vn.frames(class_id)})
                             for class_id in vn.classids(word))

                if verbs:
                    t['verbFrames'] = verbs
            except:
                pass

            # framenet
            try:
                frame_net = {}
                frames = invoke_frame(word)
                if frames is not None:
                    for fr in frames:
                        lu_temp = []
                        for lu in fn.lus(r'(?i)' + word.lower()):
                            fr_ = fn.frames_by_lemma(r'(?i)' + lu.name)
                            if len(fr_):
                                if fr_[0] == fr:
                                    lu_temp.append({'name': lu.name,
                                                    'definition': lu.definition,
                                                    'pos': lu.name.split('.')[1]})
                        frame_net[fr.ID] = {
                            'name': fr.name,
                            'frameId': fr.ID,
                            'definition': fr.definition,
                            # 'relations':fr.frameRelations,
                            'lu': lu_temp
                        }
                if frame_net:
                    t['frames'] = frame_net
            except:
                pass

        return remove_empty_fields(j)


if __name__ == "__main__":
    import json

    print(json.dumps(NltkPipeline().process(
        text="I threw the ball.",
        lang='en'),
        indent=2))
