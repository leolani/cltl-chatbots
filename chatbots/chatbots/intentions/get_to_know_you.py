import pickle
import time

from cltl.brain.long_term_memory import LongTermMemory
from emissor.persistence.persistence import ScenarioController
from emissor.representation.scenario import ImageSignal, Scenario, TextSignal

import chatbots.util.capsule_util as c_util
import chatbots.util.driver_util as d_util

### Function that tries to get the name for a new person. A while is used till the user is happy.
### From the name a unique ID *human_id* is created by adding the time_stamp
### We store the face embeddings in the friend_embeddings folder using the unique ID *human_id*
### The function returns the human_name and the human_id.
### Human_name is used to address the user, and human_id is used to store properties of the user


def get_a_name_and_id(scenario: Scenario, agent: str):
    confirm = ""
    human_name = ""
    human_id = ""
    while confirm.lower().find("yes") == -1:
        ### We take the response from the user and store it as a text signal
        utterance = input("\n")
        textSignal = d_util.create_text_signal(scenario, utterance)
        scenario.append_signal(textSignal)
        print(utterance)
        #### We hack the response to find the name of our new fiend
        #### This name needs to be set in the scenario and assigned to the global variable human
        human_name = " ".join([foo.title() for foo in utterance.strip().split()])
        human_name = "_".join(human_name.split())

        response = f"So your name is {human_name}?"
        print(f"{agent}: {response}")
        textSignal = d_util.create_text_signal(scenario, response)
        scenario.append_signal(textSignal)

        ### We take the response from the user and store it as a text signal
        confirm = input("\n")
        textSignal = d_util.create_text_signal(scenario, confirm)
        scenario.append_signal(textSignal)

    current_time = int(time.time() * 1e3)
    human_id = f"{human_name}_t_{current_time}"

    return human_name, human_id


def get_to_know_person(
    scenario_ctrl: ScenarioController,
    agent: str,
    gender: str,
    age: str,
    uuid_name: str,
    embedding,
    friends_path: str,
):
    ### This is a stranger
    ### We create the agent response and store it as a text signal
    human_name = "Stranger"
    response = (
        f"Hi there. We haven't met. I only know that \n"
        f"your estimated age is {age} \n and that your estimated gender is "
        f"{gender}. What's your name?"
    )
    print(f"{agent}: {response}")
    textSignal = d_util.create_text_signal(scenario_ctrl, response)
    scenario_ctrl.append_signal(textSignal)

    human_name, human_id = get_a_name_and_id(scenario_ctrl, agent)
    human_id = human_name  ### Hack because we cannot force the namespace through capsules, name and identity are the same till this is fixed

    #### We create the embedding
    to_save = {"uuid": uuid_name["uuid"], "embedding": embedding}
    new_friends_embedding_path = friends_path + f"/{human_id}.pkl"
    # print("New friend embedding file:",new_friends_embedding_path)
    with open(new_friends_embedding_path, "wb") as stream:
        pickle.dump(to_save, stream)

    ### The system responds to the processing of the new name input and stores it as a textsignal
    response = f"Nice to meet you, {human_name}"
    print(f"{agent}: {response}\n")
    textSignal = d_util.create_text_signal(scenario_ctrl, response)
    scenario_ctrl.append_signal(textSignal)

    return human_id, human_name, textSignal


### Function that creates capsules for the basic properties of a friend: name, age and gender.
### The capsules are sent to the BRAIN. Thoughts are caught and returned for each property.


def add_new_name_age_gender_to_brain(
    scenario_ctrl: Scenario,
    place_id: str,
    location: str,
    human_id: str,
    textSignal: TextSignal,
    imageSignal: ImageSignal,
    age: str,
    gender: str,
    human_name: str,
    my_brain: LongTermMemory,
):
    age_thoughts = ""
    gender_thoughts = ""
    name_thoughts = ""

    if human_name:
        # A triple was extracted so we compare it elementwise
        capsule = c_util.scenario_utterance_to_capsule(
            scenario_ctrl,
            place_id,
            location,
            textSignal,
            human_id,
            human_id,
            "label",
            human_name,
        )

        name_thoughts = my_brain.update(capsule, reason_types=True, create_label=True)
        print("Name capsule:", capsule)

    if age:
        # A triple was extracted so we compare it elementwise
        capsule = c_util.scenario_image_triple_to_capsule(
            scenario_ctrl,
            place_id,
            location,
            imageSignal,
            "front_camera",
            human_id,
            "age",
            str(age),
        )

        age_thoughts = my_brain.update(capsule, reason_types=True, create_label=False)
        print("Age capsule:", capsule)

    if gender:
        capsule = c_util.scenario_image_triple_to_capsule(
            scenario_ctrl,
            place_id,
            location,
            imageSignal,
            "front_camera",
            human_id,
            "gender",
            gender,
        )

        gender_thoughts = my_brain.update(capsule, reason_types=True, create_label=True)
        print("Gender capsule:", capsule)

    return name_thoughts, age_thoughts, gender_thoughts


def add_new_name_to_brain(
    scenario: ScenarioController,
    place_id: str,
    location: str,
    human_id: str,
    textSignal: TextSignal,
    human_name: str,
    my_brain: LongTermMemory,
):
    name_thoughts = ""

    # A triple was extracted so we compare it elementwise
    capsule = c_util.scenario_utterance_to_capsule(
        scenario,
        place_id,
        location,
        textSignal,
        human_id,
        human_id,
        "label",
        human_name,
    )

    name_thoughts = my_brain.update(capsule, reason_types=True, create_label=False)
    print("Name capsule:", capsule)

    return capsule, name_thoughts
