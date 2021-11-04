import os
import sys
from random import choice

from cltl.brain.long_term_memory import LongTermMemory
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.combot.backend.api.discrete import UtteranceType
from cltl.reply_generation.data.sentences import ELOQUENCE
from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.triple_extraction.api import Chat, UtteranceHypothesis
from emissor.representation.scenario import TextSignal, Scenario

src_path = os.path.abspath(os.path.join('../..'))
if src_path not in sys.path:
    sys.path.append(src_path)

import chatbots.util.capsule_util as c_util


def process_text_and_think(scenario: Scenario,
                           place_id: str,
                           location: str,
                           textSignal: TextSignal,
                           human_id: str,
                           my_brain: LongTermMemory,
                           replier: LenkaReplier):
    chat = Chat(human_id)
    chat.add_utterance([UtteranceHypothesis(c_util.seq_to_text(textSignal.seq), 1.0)])
    chat.last_utterance.analyze()
    # No triple was extracted, so we missed three items (s, p, o)

    if chat.last_utterance.triple is None:
        reply = "Any gossip?" + '\n'
    else:
        # A triple was extracted so we compare it elementwise
        capsule = c_util.scenario_utterance_and_triple_to_capsule(scenario,
                                                                  place_id,
                                                                  location,
                                                                  textSignal,
                                                                  human_id,
                                                                  chat.last_utterance.type,
                                                                  chat.last_utterance.perspective,
                                                                  chat.last_utterance.triple)

        response = my_brain.update(capsule, reason_types=True, create_label=False)
        response_json = brain_response_to_json(response)

        if replier:
            reply = replier.reply_to_statement(response_json, proactive=True, persist=False)
        else:
            reply = "Any gossip?" + '\n'

    return reply


def process_text_and_reply(scenario: Scenario,
                           place_id: str,
                           location: str,
                           human_id: str,
                           textSignal: TextSignal,
                           chat: Chat,
                           replier: LenkaReplier,
                           my_brain: LongTermMemory):
    reply = None

    chat.add_utterance([UtteranceHypothesis(c_util.seq_to_text(textSignal.seq), 1.0)])
    chat.last_utterance.analyze()

    if chat.last_utterance.triple is None:
        reply = "Sorry, did not get that."

    else:
        capsule = c_util.scenario_utterance_and_triple_to_capsule(scenario,
                                                                  place_id,
                                                                  location,
                                                                  textSignal,
                                                                  human_id,
                                                                  chat.last_utterance.type,
                                                                  chat.last_utterance.perspective,
                                                                  chat.last_utterance.triple)

        if chat.last_utterance.type == UtteranceType.QUESTION:
            response = my_brain.query_brain(capsule)
            response_json = brain_response_to_json(response)
            reply = replier.reply_to_question(response_json)

        if chat.last_utterance.type == UtteranceType.STATEMENT:
            response = my_brain.update(capsule, reason_types=True, create_label=False)
            response_json = brain_response_to_json(response)
            reply = replier.reply_to_statement(response_json, proactive=True, persist=True)

    if reply is None:
        reply = choice(ELOQUENCE)

    return reply
