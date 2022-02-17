import os
from typing import Tuple
import uuid
import time
from emissor.persistence import ScenarioStorage
from emissor.representation.scenario import ImageSignal, TextSignal, Scenario, Modality, Mention, Annotation
from emissor.representation.annotation import AnnotationType, Token, NER
from emissor.representation.container import Index

def relative_path(modality: Modality, file_name: str) -> str:
    rel_path = os.path.join(modality.name.lower())

    return os.path.join(rel_path, file_name) if file_name else rel_path


def absolute_path(scenario_storage: ScenarioStorage, scenario_id: str, modality: Modality, file_name: str = None) -> str:
    abs_path = os.path.abspath(os.path.join(scenario_storage.base_path, scenario_id, modality.name.lower()))

    return os.path.join(abs_path, file_name) if file_name else abs_path


def create_image_signal(scenario: Scenario, file: str, bounds: Tuple[int, int, int, int], timestamp: int = None):
    timestamp = int(time.time() * 1e3) if timestamp is None else timestamp
    file_path = relative_path(Modality.IMAGE, file)

    return ImageSignal.for_scenario(scenario.id, timestamp, timestamp, file_path, bounds, [])


def create_text_signal(scenario: Scenario, utterance: str, timestamp: int = None):
    timestamp = int(time.time() * 1e3) if timestamp is None else timestamp
    return TextSignal.for_scenario(scenario.id, timestamp, timestamp, [], utterance, [])

def create_text_signal_with_speaker_annotation(scenario: Scenario, utterance: str, speaker:str, timestamp: int = None):
    timestamp = int(time.time() * 1e3) if timestamp is None else timestamp
    textSignal= TextSignal.for_scenario(scenario.id, timestamp, timestamp, [], utterance, [])
    add_speaker_annotation(textSignal, speaker)
    return textSignal


def create_scenario(scenarioPath: str, scenarioid: str):
    storage = ScenarioStorage(scenarioPath)

    os.makedirs(absolute_path(storage, scenarioid, Modality.IMAGE))
    # Not yet needed
    # os.makedirs(absolut_path(storage, scenarioid, Modality.TEXT))

    print(f"Directories for {scenarioid} created in {storage.base_path}")

    return storage

def add_speaker_annotation(signal: TextSignal, speaker: str):

    current_time = int(time.time() * 1e3)
    ## AnnotationType.UTTERANCE.name.lower() #### @TODO
    annotation = [Annotation("utterance", signal.id, speaker, current_time)]
 
    signal.mentions.extend([Mention(str(uuid.uuid4()), [signal.id], [annotation])])
 