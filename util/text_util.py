
#### Example of an annotation function that adds annotations to a Signal
#### It adds NERC annotations to the TextSignal and returns a list of entities detected
import spacy
from emissor.representation.scenario import TextSignal, Mention, Annotation, Scenario


def add_ner_annotation_with_spacy(signal: TextSignal, nlp):
    processor_name = "spaCy"
    utterance = ''.join(signal.seq)

    doc = nlp(utterance)

    offsets, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                            for token in doc])

    ents = [NER.for_string(ent.label_) for ent in doc.ents]
    entity_list = [ent.text for ent in doc.ents]
    segments = [token.ruler for token in tokens if token.value in entity_list]

    annotations = [Annotation(AnnotationType.TOKEN.name.lower(), token, processor_name, int(time.time()))
                   for token in tokens]
    ner_annotations = [Annotation(AnnotationType.NER.name.lower(), ent, processor_name, int(time.time()))
                       for ent in ents]

    signal.mentions.extend([Mention(str(uuid.uuid4()), [offset], [annotation])
                            for offset, annotation in zip(offsets, annotations)])
    signal.mentions.extend([Mention(str(uuid.uuid4()), [segment], [annotation])
                            for segment, annotation in zip(segments, ner_annotations)])
    #print(entity_list)
    return entity_list

