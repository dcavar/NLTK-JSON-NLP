# NLTK-JSON-NLP

(C) 2019 by [Damir Cavar], [Oren Baldinger], Maanvitha Gongalla, Anurag Kumar, Murali Kammili, Boli Fang

Brought to you by the [NLP-Lab.org]!


## Introduction

[NLTK] wrapper to [JSON-NLP]. [NLTK] has a wide variety of capabilities, but for our purposes
we are limiting it to [WordNet], [VerbNet], and [FrameNet]. Other packages such as [spaCy] and 
[Flair] are more accurately able to annotate things like part of speech tags and dependency 
parses. See below for instruction on how to unify outputs from multiple packages.

## Microservice

The [JSON-NLP] repository provides a Microservice class, with a pre-built implementation of [Flask]. To run it, execute:
    
    python nltkjsonnlp/server.py
 
Since `server.py` extends the [Flask] app, a WSGI file would contain:

    from nltkjsonnlp.server import app as application
    
## Pipeline

[JSON-NLP] provides a simple `Pipeline` interface that we implement as `NltkPipeline`:
    
    pipeline = nltkjsonnlp.NltkPipeline()
    print(pipeline.process(text='I am a sentence.'))
            
## Unification

To make the best use of this pipeline, it is best to unify it with a more accurate and complete
pipeline such as [spaCy-NLP-Json]:

    class UnifiedPipeline(pyjsonnlp.pipeline.Pipeline):
        def __init__(self):
            super(UnifiedPipeline, self).__init__()
            self.spacy = spacynlpjson.SpacyPipeline()
            self.nltk = nltkjsonnlp.NltkPipeline()
    
        def process(self, text='', coreferences=True, constituents=False, dependencies=True, expressions=True,
                    **kwargs) -> OrderedDict:
            # start with a spacy parse
            spacy_json = self.spacy.process(text, spacy_model='en_core_web_md', constituents=False,
                                            coreferences=coreferences, dependencies=dependencies, expressions=False)
            # the get an nltk parse
            nltk_json = self.nltk.process(text)
            
            # unify the parses
            return pyjsonnlp.unification.unifier.add_annotation_to_a_from_b(a=spacy_json, 
                                                                            b=nltk_json, annotation='tokens')



[Damir Cavar]: http://damir.cavar.me/ "Damir Cavar"
[Oren Baldinger]: https://oren.baldinger.me/ "Oren Baldinger"
[NLP-Lab.org]: http://nlp-lab.org/ "NLP-Lab.org"
[JSON-NLP]: https://github.com/dcavar/JSON-NLP "JSON-NLP"
[Flair]: https://github.com/zalandoresearch/flair "Flair"
[spaCy]: https://spacy.io/ "spaCy"
[NLTK]: http://nltk.org/ "Natural Language Processing Toolkit"
[Polyglot]: https://github.com/aboSamoor/polyglot "Polyglot" 
[Xrenner]: https://github.com/amir-zeldes/xrenner "Xrenner"
[CONLL-U]: https://universaldependencies.org/format.html "CONLL-U"
[spaCy-NLP-Json]: https://github.com/dcavar/spaCy-JSON-NLP "spaCy-JSON-NLP"
[WordNet]: https://wordnet.princeton.edu/ "Wordnet"
[FrameNet]: https://framenet.icsi.berkeley.edu/fndrupal/ "FrameNet"
[VerbNet]: https://verbs.colorado.edu/~mpalmer/projects/verbnet.html "VerbNet"
