
from transformers import pipeline, AutoTokenizer
import re

class USR_MLM:
    def __init__(self, path=None, top_results=20):
        """ Load pretrained RoBERTa model for masked Langauge model based likelihood.

            params
            str path: path to stored model or None

            returns: None
        """
        if path is None:
            self.__model_name = 'adamlin/usr-topicalchat-roberta_ft'
        else:
            self.__model_name = path

        self.__tokenizer = AutoTokenizer.from_pretrained(self.__model_name)
        self.__model = pipeline("fill-mask", model=self.__model_name)
        self.__model.top_k = top_results  ### we check against the top results


    def mask_target_sentence(self, context, target):
        masked_targets = []
        target_tokens = re.split(' ', target)
        for index, token in enumerate(target_tokens):
            sequence = context+" "
            for token in target_tokens[:index]:
                sequence+= token+" "
            sequence += self.__tokenizer.mask_token
            for token in target_tokens[index+1:]:
                sequence+= " "+token
            masked_targets.append(sequence)
        return masked_targets, target_tokens


    def sentence_likelihood(self, context, target):
        masked_targets, target_tokens = self.mask_target_sentence(context, target)
        expected_target = ""
        max_scores = []
        scores = []
        for masked_target, token in zip(masked_targets, target_tokens):
            results = self.__model(masked_target)
            expected_target += results[0]['token_str'] + " "
            max_scores.append(results[0]['score'])
            match = False
            for result in results:
                if result['token_str'].lower().strip() == token.lower():
                    scores.append(result['score'])
                    match = True
                    break

            if not match:
                scores.append(0)
        likelihood = sum(scores) / len(scores)
        max_likelihood = sum(max_scores) / len(max_scores)

        return likelihood, expected_target, max_likelihood

    def score_pairs_for_likelihood(self, turns:[]):
        for context, target in turns:
            llh, best_sentence, max_score = self.sentence_likelihood(self, context, target)
            print(turn)
            print('Likelihood:', llh, 'Max score:', max_score, 'Best sentence:', best_sentence)

if __name__ == "__main__":
    turns = [('Do you have a cat?', 'I do not have a cat'),  # good
             ('Do you have a cat?', 'I like cats'),  # not as good
             ('Do you have a cat?', 'I like kittens'),  # worse
             ('Do you have a cat?', 'I want a turtle')]  # what are we even saying
    ###### Likelihood
    top_results = 20
    model_path = 'adamlin/usr-topicalchat-roberta_ft'
    model_path = 'xlm-roberta-base'
    model_path = 'roberta-base'
    model_mlm = USR_MLM(model_path, top_results)
    for context, target in turns:
        llh, best_sentence, max_score = model_mlm.sentence_likelihood(context, target)
        print(turn)
        print('Likelihood:', llh, 'Max score:', max_score, 'Best sentence:', best_sentence)
