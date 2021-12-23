#### Example of an annotation function that adds annotations to a Signal
#### It adds NERC annotations to the TextSignal and returns a list of entities detected

import uuid

import time
from emissor.representation.annotation import AnnotationType, Token, NER
from emissor.representation.container import Index
from emissor.representation.scenario import TextSignal, Mention, Annotation


def add_ner_annotation_with_spacy(signal: TextSignal, nlp):
    processor_name = "spaCy"
    utterance = ''.join(signal.seq)

    doc = nlp(utterance)

    offsets, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                            for token in doc])

    ents = [NER.for_string(ent.label_) for ent in doc.ents]
    entity_list = [ent.text for ent in doc.ents]
    segments = [token.ruler for token in tokens if token.value in entity_list]

    current_time = int(time.time() * 1e3)
    annotations = [Annotation(AnnotationType.TOKEN.name.lower(), token, processor_name, current_time)
                   for token in tokens]
    ner_annotations = [Annotation(AnnotationType.NER.name.lower(), ent, processor_name, current_time)
                       for ent in ents]

    signal.mentions.extend([Mention(str(uuid.uuid4()), [offset], [annotation])
                            for offset, annotation in zip(offsets, annotations)])
    signal.mentions.extend([Mention(str(uuid.uuid4()), [segment], [annotation])
                            for segment, annotation in zip(segments, ner_annotations)])
    # print(entity_list)
    return entity_list



def add_np_annotation_with_spacy(signal: TextSignal, nlp):

    rels={'nsubj', 'dobj', 'prep'}
    """
    extract predicates with:
    -subject
    -object
    
    :param spacy.tokens.doc.Doc doc: spaCy object after processing text
    
    :rtype: list 
    :return: list of tuples (predicate, subject, object)
    """
    processor_name = "spaCy"
    utterance = ''.join(signal.seq)

    doc = nlp(utterance)
    offsets, tokens = zip(*[(Index(signal.id, token.idx, token.idx + len(token)), Token.for_string(token.text))
                            for token in doc])

    
    predicates = {}
    subjects_and_objects = []
    
    for token in doc:
        if token.dep_ in rels:
            
            head = token.head
            head_id = head.i
            
            if head_id not in predicates:
                predicates[head_id] = dict()
            
            subjects_and_objects.append(token.lemma_)
            
            predicates[head_id][token.dep_] = token.lemma_
   #### Change this to create triples 
    #output = []
    #for pred_token, pred_info in predicates.items():
    #    one_row = (doc[pred_token].lemma_, 
    #               pred_info.get('nsubj', None),
    #               pred_info.get('dobj', None)
    #              )
    #    output.append(one_row)

    segments = [token.ruler for token in tokens if token.value in subjects_and_objects]

    current_time = int(time.time() * 1e3)
    annotations = [Annotation(AnnotationType.TOKEN.name.lower(), token, processor_name, current_time)
                   for token in tokens]

    signal.mentions.extend([Mention(str(uuid.uuid4()), [offset], [annotation])
                            for offset, annotation in zip(offsets, annotations)])
    # print(subjects_and_objects)
    return subjects_and_objects
