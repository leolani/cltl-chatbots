# general imports for EMISSOR and the BRAIN
import emissor as em
from cltl import brain
from cltl.triple_extraction.api import Chat, UtteranceHypothesis
from emissor.persistence import ScenarioStorage
from emissor.representation.scenario import Modality, ImageSignal, TextSignal, Mention, Annotation, Scenario
from cltl.brain.long_term_memory import LongTermMemory
from cltl.combot.backend.api.discrete import UtteranceType
from cltl.reply_generation.lenka_replier import LenkaReplier

# specific imports
import spacy
from datetime import datetime
import requests
import pathlib

import driver_util as d_util
import capsule_util as c_util
import text_util as t_util
import text_to_triple as ttt


### Load a language model in spaCy
nlp = spacy.load('en_core_web_sm')


log_path=pathlib.Path('./logs')
print(type(log_path))
my_brain = brain.LongTermMemory(address="http://localhost:7200/repositories/sandbox",
                           log_dir=log_path,
                           clear_all=True)