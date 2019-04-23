# /usr/bin/env python3
"""
(C) 2019 Damir Cavar, Oren Baldinger, Maanvitha Gongalla, Anurag Kumar, Murali Kammili, Boli Fang

Testing wrappers for NLTK to JSON-NLP output format.

Licensed under the Apache License 2.0, see the file LICENSE for more details.

Brought to you by the NLP-Lab.org (https://nlp-lab.org/)!
"""

from collections import OrderedDict
from typing import Callable
from unittest import TestCase

import pyjsonnlp
from pyjsonnlp import validation

import nltkjsonnlp
from nltkjsonnlp import NltkPipeline

text = "I went to the bank."


class TestNltkPipeline(TestCase):
    def test_process(self):
        pyjsonnlp.__version__ = '0.2.9'
        actual = NltkPipeline.process(text)
        assert isinstance(actual, OrderedDict)

    def test_get_wordnet_pos(self):
        actual = nltkjsonnlp.get_wordnet_pos('JJ')
        expected = nltkjsonnlp.wordnet.ADJ
        assert expected == actual, actual
        actual = nltkjsonnlp.get_wordnet_pos('VV')
        expected = nltkjsonnlp.wordnet.VERB
        assert expected == actual, actual
        actual = nltkjsonnlp.get_wordnet_pos('RR')
        expected = nltkjsonnlp.wordnet.ADV
        assert expected == actual, actual
        actual = nltkjsonnlp.get_wordnet_pos('anything else')
        expected = nltkjsonnlp.wordnet.NOUN
        assert expected == actual, actual

    def test_get_lemmatizer(self):
        t = nltkjsonnlp.get_lemmatizer()
        assert isinstance(t, Callable), t

    def test_get_stemmer(self):
        t = nltkjsonnlp.get_stemmer()
        assert isinstance(t, Callable), t

    def test_validation(self):
        assert validation.is_valid(NltkPipeline.process(text, lang='en'))
