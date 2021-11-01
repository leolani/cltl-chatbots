import emissor as em
from cltl import brain
from cltl.triple_extraction.api import Chat, UtteranceHypothesis
from emissor.persistence import ScenarioStorage
from emissor.representation.annotation import AnnotationType, Token, NER
from emissor.representation.container import Index
from emissor.representation.scenario import Modality, ImageSignal, TextSignal, Mention, Annotation, Scenario
from cltl.brain.long_term_memory import LongTermMemory
from cltl.combot.backend.api.discrete import UtteranceType
import sys
import os
src_path = os.path.abspath(os.path.join('..'))
if src_path not in sys.path:
    sys.path.append(src_path)

import util.driver_util as d_util
import util.capsule_util as c_util
import util.face_util as f_util


def process_text_and_think (scenario: Scenario, 
                  place_id:str, 
                  location: str, 
                  textSignal: TextSignal,
                  human_id: str,
                  my_brain:LongTermMemory):
    thoughts = ""
    chat = Chat(human_id)
    chat.add_utterance([UtteranceHypothesis(c_util.seq_to_text(textSignal.seq), 1.0)])
    chat.last_utterance.analyze()
    # No triple was extracted, so we missed three items (s, p, o)
    if chat.last_utterance.triple is None:
        utterance = "Any gossip?" + '\n'
    else:
        triple = c_util.rephrase_triple_json_for_capsule(chat.last_utterance.triple)
        # A triple was extracted so we compare it elementwise
        capsule = c_util.scenario_utterance_and_triple_to_capsule(scenario, 
                                                                  place_id,
                                                                  location,
                                                                  textSignal, 
                                                                  human_id,
                                                                  chat.last_utterance.perspective, 
                                                                  triple)
        print('Capsule:', capsule)
        thoughts = my_brain.update(capsule, reason_types=True)
        #print(thoughts)
    return thoughts    
