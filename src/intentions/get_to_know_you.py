import emissor as em
from cltl import brain
from cltl.triple_extraction.api import Chat, UtteranceHypothesis
from emissor.persistence import ScenarioStorage
from emissor.representation.annotation import AnnotationType, Token, NER
from emissor.representation.container import Index
from emissor.representation.scenario import Modality, ImageSignal, TextSignal, Mention, Annotation, Scenario
from cltl.brain.long_term_memory import LongTermMemory
from cltl.combot.backend.api.discrete import UtteranceType

### Function that tries to get the name for a new person. A while is used till the user is happy.
### From the name a unique ID *human_id* is created by adding the time_stamp
### We store the face embeddings in the friend_embeddings folder using the unique ID *human_id*
### The function returns the human_name and the human_id.
### Human_name is used to address the user, and human_id is used to store properties of the user

def get_to_know_person(scenario: Scenario, agent:str, gender:str, age: str, uuid_name: str, embedding):
        ### This is a stranger
        ### We create the agent response and store it as a text signal
        human_name = "Stranger"
        response = (
            f"Hi there. We haven't met. I only know that \n"
            f"your estimated age is {age} \n and that your estimated gender is "
            f"{gender}. What's your name?"
        )
        print(f"{agent}: {response}")
        textSignal = d_util.create_text_signal(scenario, response)
        scenario.append_signal(textSignal)
        
        confirm = ""
        while confirm.lower().find("yes")==-1:
            ### We take the response from the user and store it as a text signal
            utterance = input("\n")
            textSignal = d_util.create_text_signal(scenario, utterance)
            scenario.append_signal(textSignal)
            print(utterance)
            #### We hack the response to find the name of our new fiend
            #### This name needs to be set in the scenario and assigned to the global variable human
            human_name = " ".join([foo.title() for foo in utterance.strip().split()])
            human_name = "_".join(human_name.split())
        
            response = (f"So your name is {human_name}?")
            print(f"{agent}: {response}")
            textSignal = d_util.create_text_signal(scenario, response)
            scenario.append_signal(textSignal)
            
            ### We take the response from the user and store it as a text signal
            confirm = input("\n")
            textSignal = d_util.create_text_signal(scenario, confirm)
            scenario.append_signal(textSignal)


        current_time = str(datetime.now().microsecond)
        human_id = human_name+"_t_"+current_time
        #### We create the embedding
        to_save = {"uuid": uuid_name["uuid"], "embedding": embedding}

        with open(f"./friend_embeddings/{human_id}.pkl", "wb") as stream:
            pickle.dump(to_save, stream)
            
        return human_id, human_name, textSignal
    


### Function that creates capsules for the basic properties of a friend: name, age and gender.
### The capsules are sent to the BRAIN. Thoughts are caught and returned for each property.

def process_new_friend_and_think (scenario: Scenario, 
                  place_id:str, 
                  location: str, 
                  human_id: str,
                  textSignal: TextSignal,
                  imageSignal: ImageSignal,
                  age: str,
                  gender: str,
                  human_name: str,
                  my_brain:LongTermMemory):
    age_thoughts = ""
    gender_thoughts = ""
    name_thoughts = ""


    if human_name:
        # A triple was extracted so we compare it elementwise
        perspective = {"certainty": 1, "polarity": 1, "sentiment": 1}
        capsule = c_util.scenario_utterance_to_capsule(scenario, 
                                                                  place_id,
                                                                  location,
                                                                  textSignal,
                                                                  human_id,
                                                                  perspective,
                                                                  human_id,
                                                                  "label",
                                                                  human_name)

        name_thoughts = my_brain.update(capsule, reason_types=True, create_label=False)
        print('Name capsule:', capsule)


    if age:
        # A triple was extracted so we compare it elementwise
        capsule = c_util.scenario_image_triple_to_capsule(scenario, 
                                                                  place_id,
                                                                  location,
                                                                  imageSignal,
                                                                  "front_camera", 
                                                                  human_id,
                                                                  "age",
                                                                  str(age))

        age_thoughts = my_brain.update(capsule, reason_types=True, create_label=False)
        print('Age capsule:', capsule)

    if gender:
        capsule = c_util.scenario_image_triple_to_capsule(scenario, 
                                                                  place_id,
                                                                  location,
                                                                  imageSignal,
                                                                  "front_camera", 
                                                                  human_id,
                                                                  "gender",
                                                                  gender)

        gender_thoughts = my_brain.update(capsule, reason_types=True, create_label=False)
        print('Gender capsule:', capsule)

    return name_thoughts, age_thoughts, gender_thoughts    
