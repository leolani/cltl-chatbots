import sys

from dialogue_evaluation_usr import USR_CTX
from dialogue_evaluation_likelihood import USR_MLM
from emissor.persistence import ScenarioStorage
from emissor.representation.scenario import Modality, TextSignal, Mention, Annotation, Scenario
import getopt


def get_context_target_pairs_from_signals(signals:[]):
    pairs = []
    context = ""
    target = ""
    for index, signal in enumerate(signals):
        if index == 0:
            target = ''.join(signal.seq)
        else:
            context += " "+target
            target = ''.join(signal.seq)
        pair = (context, target)
        pairs.append(pair)
    return pairs

if __name__ == "__main__":

    metric=None
    top=20
    scenario_path=None
    scenario_id=None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm:t:p:s:", ["metric=", "scenario-path=", "scenario-name", "top-result"])
    except getopt.GetoptError:
        print('Usage:', 'evaluate_emissor_dialogue -p <scenario-path> -s  <scenario_name> -m <metric>')
        print('Metric values:\n\tCTX = USR Context', '\n\tUK= USR use of knowledge',
              '\n\tMLM= Likelikehood by Masked Language Model', '\n\tALL= all three scores')
        sys.exit(2)
    if len(sys.argv)==1:
        print('Usage:', 'evaluate_emissor_dialogue -p <scenario-path> -s  <scenario_name> -m <metric>')
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

    ### Create the scenario folder, the json files and a scenarioStorage and scenario in memory
    scenarioStorage = ScenarioStorage(scenario_path)
    scenario_ctrl = scenarioStorage.load_scenario(scenario_id)
    signals = scenario_ctrl.get_signals(Modality.TEXT)

    pairs = get_context_target_pairs_from_signals(signals)
    print('Nr of pairs:', len(pairs), ' extracted from scenario: ', scenario_id)

    if metric=="CTX" or metric=="ALL":
        scores = []
        model_path_ctx = 'adamlin/usr-topicalchat-ctx'
        model_ctx = USR_CTX(path=model_path_ctx)
        for context, response in pairs:
            score = model_ctx.MCtx(context, response)
            print('Context score:', score, '\t', context, response)
            scores.append(score)
        average_score = sum(scores)/len(scores)
        print('\nAverage score:', average_score)
        print('Sequence profile:', scores)

    if metric=="UK" or metric=="ALL":
        scores = []
        model_path_uk = 'adamlin/usr-topicalchat-uk'
        model_uk = USR_CTX(path=model_path_uk)
        for context, response in pairs:
            score = model_uk.MCtx(context, response)
            print('Context score:', score, '\t', context, response)
            scores.append(score)
        average_score = sum(scores)/len(scores)
        print('\nAverage score:', average_score)
        print('Sequence profile:', scores)

    if metric=="MLM" or metric=="ALL":
        scores = []
        max_scores=[]
        model_path_mlm = 'adamlin/usr-topicalchat-roberta_ft'
        model_mlm = USR_MLM(path=model_path_mlm, top_results=top)
        for pair in pairs:
            llh, best_sentence, max_score = model_mlm.sentence_likelihood(pair)
            print(pair)
            print('Likelihood:', llh, '\t', pair[0], pair[1] )
            print('\tMax score:', max_score, 'Best sentence:', best_sentence)
            scores.append(llh)
            max_scores.append(max_score)

        average_score = sum(scores)/len(scores)
        print('\nAverage score:', average_score)
        print('Sequence profile:', scores)

        average_max_score = sum(max_scores)/len(max_scores)
        print('\nAverage max score:', average_max_score)
        print('Sequence max profile:', max_scores)

