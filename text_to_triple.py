##### Some dummy function to generate triples from text and lists of entities, this needs to be replaced by something serious
from cltl.brain.long_term_memory import LongTermMemory


def getTriplesFromEntities(entityList, mention:str):
    subj = ""
    pred = "relatesTo"
    obj = ""
    for ent1 in entityList:
        if len(entityList)==1:
            subj = ent1
            pred="denotedBy"
            obj= mention
        else:
            for ent2 in entityList:
                if not ent1==ent2:
                    subj = ent1
                    obj = ent2    
    return subj, pred, obj


def getTextFromTriples (response):
    utterance = ""
    if response:
        thought = response['thoughts']
        if thought:
            utterance="I am thinking: "
            if thought._statement_novelty:
                print(type(thought._statement_novelty))
                for item in thought._statement_novelty:
                    utterance+= item
                utterance+='; '
            if thought._complement_conflict:
                for item in thought._complement_conflict:
                    utterance+= item
                utterance+='; '
            if thought._negation_conflicts:
                for item in thought._negation_conflicts:
                    utterance+= item
                utterance+='; '

            if thought._entity_novelty:
                utterance+= str(thought._entity_novelty)
                utterance+='; '
            if thought._complement_gaps:
                utterance += str(thought._complement_gaps)
                utterance+='; '
            if thought._overlaps:
                utterance += str(thought._overlaps)
                utterance+=';'
            if thought._subject_gaps:
                utterance += str(thought._subject_gaps)
                utterance+='; '
    return utterance