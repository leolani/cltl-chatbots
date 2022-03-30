# general imports for EMISSOR and the BRAIN

import pathlib
import time
# specific imports
from random import choice, getrandbits

import spacy
from cltl import brain
from cltl.reply_generation.data.sentences import (ASK_NAME, ELOQUENCE,
                                                  GREETING, TALK_TO_ME)
from cltl.reply_generation.lenka_replier import LenkaReplier
from cltl.triple_extraction.api import Chat
from cltl.triple_extraction.cfg_analyzer import CFGAnalyzer
from emissor.representation.scenario import (Annotation, ImageSignal, Mention,
                                             Modality, Scenario, TextSignal)

import chatbots.chatbots.intentions.talk as talk
import chatbots.chatbots.util.capsule_util as c_util
#### The next utils are needed for the interaction and creating triples and capsules
import chatbots.chatbots.util.driver_util as d_util
import chatbots.chatbots.util.face_util as f_util
import chatbots.chatbots.util.text_util as t_util
from chatbots.chatbots.bots import emissor_api


def run_docker_containers():
    containers = []

    container_erc = f_util.start_docker_container("tae898/emoberta-large", 10006)
    containers.append(container_erc)

    return containers


def kill_docker_containers(containers):

    for container in containers:
        f_util.kill_container(container)


def get_subj_obj_labels_from_capsules(capsule_list):
    mentions = []
    for capsule in capsule_list:
        label = capsule["subject"]["label"]
        if label:
            mentions.append(label)
        label = capsule["object"]["label"]
        if label:
            mentions.append(label)
    mentions = list(set(mentions))
    return mentions


def add_mention_to_episodic_memory(
    textSignal: TextSignal,
    source,
    mention_list,
    my_brain,
    scenario_ctrl,
    location,
    place_id,
):
    response_list = []
    for mention in mention_list:

        ### We created a perceivedBy triple for this experience,
        ### @TODO we need to include the bouding box somehow in the object
        # print(mention)
        capsule = c_util.scenario_image_triple_to_capsule(
            scenario_ctrl,
            textSignal,
            location,
            place_id,
            source,
            mention,
            "denotedIn",
            textSignal.id,
        )

        # print(capsule)
        # Create the response from the system and store this as a new signal
        # We use the throughts to respond
        response = my_brain.update(capsule, reason_types=True, create_label=True)
        response_list.append(response)
    return response_list


def listen_and_remember(
    scenario_ctrl, AGENT, HUMAN_NAME, HUMAN_ID, my_brain, location, place_id
):
    print_details = False
    replier = LenkaReplier()
    analyzer = CFGAnalyzer()
    chat = Chat(HUMAN_ID)
    #### Initial prompt by the system from which we create a TextSignal and store it
    initial_prompt = f"{choice(TALK_TO_ME)}"
    print(AGENT + ": " + initial_prompt)
    textSignal = d_util.create_text_signal(scenario_ctrl, initial_prompt)
    scenario_ctrl.append_signal(textSignal)

    utterance = ""
    #### Get input and loop
    while not (utterance.lower() == "stop" or utterance.lower() == "bye"):
        ###### Getting the next input signals
        utterance = input("\n")
        emotion = t_util.recognize_emotion(utterance)
        print(HUMAN_NAME + ": " + utterance)
        textSignal = d_util.create_text_signal(scenario_ctrl, utterance)
        utterance_timestamp = int(time.time() * 1e3)
        mention = t_util.create_emotion_mention(
            textSignal, "machine", utterance_timestamp, emotion
        )
        textSignal.mentions.append(mention)
        scenario_ctrl.append_signal(textSignal)

        #### Process input and generate reply

        capsule_list, reply_list, response_list = talk.process_statement_and_reply(
            scenario_ctrl,
            place_id,
            location,
            HUMAN_ID,
            textSignal,
            chat,
            analyzer,
            replier,
            my_brain,
            print_details,
        )

        reply = ""
        for a_reply in reply_list:
            reply += a_reply + ". "
        print(AGENT + ": " + reply)
        textSignal = d_util.create_text_signal(scenario_ctrl, reply)
        scenario_ctrl.append_signal(textSignal)

        ###### Add denotedIn links for every subject and object label
        mention_list = get_subj_obj_labels_from_capsules(capsule_list)
        add_mention_to_episodic_memory(
            textSignal,
            HUMAN_ID,
            mention_list,
            my_brain,
            scenario_ctrl,
            location,
            place_id,
        )


def main():
    containers = run_docker_containers()
    ### Link your camera
    nlp = spacy.load("en_core_web_sm")
    # Initialise the brain in GraphDB

    ##### Setting the agents
    AGENT = "Leolani2"
    HUMAN_NAME = "Stranger"
    HUMAN_ID = "stranger1"
    (
        scenarioStorage,
        scenario_ctrl,
        imagefolder,
        rdffolder,
        location,
        place_id,
    ) = emissor_api.start_a_scenario(AGENT, HUMAN_ID, HUMAN_NAME)

    log_path = pathlib.Path(rdffolder)
    my_brain = brain.LongTermMemory(
        address="http://localhost:7200/repositories/sandbox",
        log_dir=log_path,
        clear_all=True,
    )

    listen_and_remember(
        scenario_ctrl, AGENT, HUMAN_NAME, HUMAN_ID, my_brain, location, place_id
    )
    scenario_ctrl.scenario.ruler.end = int(time.time() * 1e3)
    scenarioStorage.save_scenario(scenario_ctrl)

    kill_docker_containers(containers)


if __name__ == "__main__":
    main()
