import sys

from dialogue_evaluation_usr import USR_CTX
from dialogue_evaluation_likelihood import USR_MLM
from emissor.persistence import ScenarioStorage
from emissor.representation.scenario import Modality, TextSignal, Mention, Annotation, Scenario
import getopt

#  THIS SCRIPT USES A REIMPLEMENTATION OF THE USR METRICS FOR EVALUATING DIALOGUES. THE BASICS ARE DESCRIBED IN THE FOLLOWING PAPER:
# Mehri, Shikib, and Maxine Eskenazi. "Usr: An unsupervised and reference free evaluation metric for dialog generation." arXiv preprint arXiv:2005.00456 (2020).
# https://arxiv.org/pdf/2005.00456.pdf
#
# OUR IMPLEMENTATION USES THE MODELS PROVIDED BY THE AUTHORS: http://shikib.com/usr AND ALSO AVAILABLE ON HUGGINGFACE.COM
# WE REIMPLEMENTED THE LML SCORE USING A MASKED LIKELIHOOD FOR THE TARGET SENTENCE.
# THIS SCRIPT LOADS A DIALIGUE FROM THE EMISSOR DATA FORMAT. THE DETAILS OF THE EMISSOR FORMAT ARE EXPLAINED HERE: https://github.com/cltl/EMISSOR
#
# YOU CAN CREATE EMISSOR DIALOGUES USING THE CHATBOT NOTEBOOKS AND SCRIPTS IN THIS REPOSITORY
#





def get_turns_with_context_from_signals(signals:[], max_context=200):
    triples = []
    context = ""
    target = ""
    cue =""
    for index, signal in enumerate(signals):
        if index == 0:
            target = ''.join(signal.seq)
        else:
            cue = target
            context += " "+target
            target = ''.join(signal.seq)
        if len(context)>max_context:
            context = context[len(context)-max_context:]
        triple = (context, target, cue)
        triples.append(triple)
    return triples

if __name__ == "__main__":

    metric=None
    top=20
    scenario_path=None
    scenario_id=None
    max_context=200

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm:t:p:s:c:", ["metric=", "scenario-path=", "scenario-name", "top-result", "max-context"])
    except getopt.GetoptError:
        print('Usage:', 'evaluate_emissor_dialogue -p <scenario-path> -s  <scenario_name> -m <metric> -c <max-context>')
        print('Metric values:\n\tCTX = USR Context', '\n\tUK= USR use of knowledge',
              '\n\tMLM= Likelikehood by Masked Language Model', '\n\tALL= all three scores')
        sys.exit(2)
    if len(sys.argv)==1:
        print('Usage:', 'evaluate_emissor_dialogue -p <scenario-path> -s  <scenario_name> -m <metric> -c <max-context>')
        print('Metric values:\n\tCTX = USR Context', '\n\tUK= USR use of knowledge',
              '\n\tMLM= Likelikehood by Masked Language Model', '\n\tALL= all three scores')
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print('Usage:', 'evaluate_emissor_dialogue -p <scenario-path> -s  <scenario_name> -m <metric>')
            print('Metric values:\n\tCTX = USR Context', '\n\tUK= USR use of knowledge',
                  '\n\tMLM= Likelikehood by Masked Language Model', '\n\tALL= all three scores')
            sys.exit()
        elif opt in ("-p", "--scenario-path"):
            print('Scenario-path:', arg)
            scenario_path = arg
        elif opt in ("-s", "--scenario-name"):
            print('Scenario-name:', arg)
            scenario_id = arg
        elif opt in ("-m", "--metric"):
            print('USR metric:', arg)
            metric = arg
        elif opt in ("-t", "--top-result"):
            print('Top-result:', arg)
            top = int(arg)
        elif opt in ("-c", "--max-context"):
            print('Max-context:', arg)
            max_context = int(arg)

    ### Create the scenario folder, the json files and a scenarioStorage and scenario in memory
    scenarioStorage = ScenarioStorage(scenario_path)
    scenario_ctrl = scenarioStorage.load_scenario(scenario_id)
    signals = scenario_ctrl.get_signals(Modality.TEXT)

    turns = get_turns_with_context_from_signals(signals, max_context)
    print('Nr of turns:', len(turns), ' extracted from scenario: ', scenario_id)

    if metric=="CTX" or metric=="ALL":
        scores = []
        even_scores=[]
        odd_scores=[]
        model_path_ctx = 'adamlin/usr-topicalchat-ctx'
        model_ctx = USR_CTX(path=model_path_ctx)
        for index, turn in enumerate(turns):
            context = turn[0]
            target = turn[1]
            cue = turn[2]
            print('Context score:', score, '\t', context, response)
            scores.append(score)
            if index % 2 == 0:
                even_scores.append(score)
            else:
                odd_scores.append(score)

        average_score = sum(scores) / len(scores)
        print('\nAverage score:', average_score)
        print('Sequence profile:', scores)

        average_even_score = sum(even_scores) / len(even_scores)
        print('\nAverage odd score:', average_even_score)
        print('Even sequence profile:', even_scores)

        average_odd_score = sum(odd_scores) / len(odd_scores)
        print('\nAverage odd score:', average_odd_score)
        print('Odd sequence profile:', odd_scores)

    if metric=="UK" or metric=="ALL":
        scores = []
        odd_scores=[]
        even_scores=[]
        model_path_uk = 'adamlin/usr-topicalchat-uk'
        model_uk = USR_CTX(path=model_path_uk)

        for index, turn in enumerate(turns):
            context = turn[0]
            target = turn[1]
            cue=turn[2]
            print('Context score:', score, '\t', context, response)
            scores.append(score)
            if index % 2 == 0:
                even_scores.append(score)
            else:
                odd_scores.append(score)

        average_score = sum(scores)/len(scores)
        print('\nAverage score:', average_score)
        print('Sequence profile:', scores)

        average_even_score = sum(even_scores) / len(even_scores)
        print('\nAverage odd score:', average_even_score)
        print('Even sequence profile:', even_scores)

        average_odd_score = sum(odd_scores) / len(odd_scores)
        print('\nAverage odd score:', average_odd_score)
        print('Odd sequence profile:', odd_scores)

    if metric=="MLM" or metric=="ALL":
        scores = []
        max_scores=[]
        odd_scores=[]
        odd_max_scores=[]
        even_scores=[]
        even_max_scores=[]

        model_path_mlm = 'adamlin/usr-topicalchat-roberta_ft'
        model_mlm = USR_MLM(path=model_path_mlm, top_results=top)
        for index, turn in enumerate(turns):
            context = turn[0]
            target = turn[1]
            cue=turn[2]
            llh, best_sentence, max_score = model_mlm.sentence_likelihood(context,target)
            print('Turn:', index)
            print('Input:', cue)
            print('Response', target)
            print('\tLikelihood:', llh)
            print('Best model response', best_sentence)
            print('\tMax score:', max_score)
            scores.append(llh)
            max_scores.append(max_score)

            if index % 2 == 0:
                even_scores.append(llh)
                even_max_scores.append(max_score)
            else:
                odd_scores.append(llh)
                odd_max_scores.append(max_score)


        average_score = sum(scores)/len(scores)
        print('\nAverage score:', average_score)
        print('Sequence profile:', scores)

        average_max_score = sum(max_scores)/len(max_scores)
        print('\nAverage max score:', average_max_score)
        print('Sequence max profile:', max_scores)

        average_even_score = sum(even_scores) / len(even_scores)
        print('\nAverage even score:', average_even_score)
        print('Even sequence profile:', even_scores)

        average_max_even_score = sum(odd_max_scores) / len(even_max_scores)
        print('\nAverage max even score:', average_max_score)
        print('Max even sequence max profile:', even_max_scores)

        average_odd_score = sum(odd_scores) / len(odd_scores)
        print('\nAverage odd score:', average_odd_score)
        print('Odd sequence profile:', odd_scores)

        average_max_odd_score = sum(odd_max_scores) / len(odd_max_scores)
        print('\nAverage max odd score:', average_max_odd_score)
        print('Max odd sequence max profile:', odd_max_scores)

