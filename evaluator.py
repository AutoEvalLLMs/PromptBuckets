from genbit.genbit_metrics import GenBitMetrics
import csv
import evaluate
from tqdm import tqdm
import json

class AutoEvalEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def load_prompts_csv(prompt_fp):
        with open(prompt_fp, "r") as f:
            ret = [p[-1] for p in csv.reader(f)]
        return ret

    def evaluate():
        raise NotImplementedError()

    def evaluate_from_generations():
        raise NotImplementedError()

    @staticmethod
    def to_json(obj, fp):
        with open("fp", "w") as f:
            json.dump(obj, f)


class GenBitWrapper():
    def __init__(self):
        """
        Genbit Evaluator
        add_data, callable: input = string -- adds string to evaluations

        evaluate: method calculates and returns all metrics split by token.
        """
        self.genbit = GenBitMetrics(language_code='en', context_window=5, distance_weight=0.95, percentile_cutoff=80)

    def add_data(self, string):
        self.genbit.add_data(string, tokenized=False)

    def evaluate(self):
        return self.genbit.get_metrics(output_statistics=True, output_word_list=True)

    def __call__(self, string):
        self.add_data(string)


class GenBitEvaluator(AutoEvalEvaluator):
    def __init__(self, prompts_fp):
        self.prompts = AutoEvalEvaluator.load_prompts_csv(prompts_fp)

    def evaluate(self, targetLLM, num_gen=1):
        generations = {}
        for p in tqdm(self.prompts):
            genbit = GenBitWrapper()
            outputs = [targetLLM.generate(p) for i in range(num_gen)]
            for t in outputs:
                genbit(t) # add to genbit
            
            generations[p] = (genbit.evaluate(), outputs)
        return generations

    def evaluate_from_generations(self, prompts, generations):
        ret = {}
        for i,p in tqdm(enumerate(prompts)):
            genbit = GenBitWrapper()
            for t in generations[i]:
                genbit(t)
            ret[p] = (genbit.evaluate(), generations[i])

        return ret


class HonestEvaluator(AutoEvalEvaluator):
    def __init__(self, prompts_fp):
        self.prompts = AutoEvalEvaluator.load_prompts_csv(prompts_fp)
        self.honest = evaluate.load("honest", "en")

    def evaluate(self, targetLLM, num_gen=1):
        generations = {}
        for p in tqdm(self.prompts):
            gens = [targetLLM.generate(p) for i in range(num_gen)]
            honest = self.honest.compute(predictions=[t.split(' ') for t in gens]) # split by word for honest
            generations[p] = (honest, gens)
        return generations

    def evaluate_from_generations(self, prompts, generations):
        ret = {}
        for i, p in tqdm(enumerate(prompts)):
            ret[p] = (self.honest.compute(predictions = [t.split(' ') for t in generations[i]]), generations[i])
        return ret


class RegardEvaluator(AutoEvalEvaluator):
    def __init__(self, prompts_fp):
        self.prompts = AutoEvalEvaluator.load_prompts_csv(prompts_fp)
        self.regard = evaluate.load("regard", module_type="measurement")

    def evaluate(self, targetLLM, num_gen=1):
        generations = {}
        for p in tqdm(self.prompts):
            generations[p] = [targetLLM.generate(p) for i in range(num_gen)]
        for k,v in generations.items():
            generations[k] = (self.regard.compute(data=v, aggregation='maximum'), v) # could do some mapping here with regard https://huggingface.co/spaces/evaluate-measurement/regard

        return generations

    def evaluate_from_generations(self, prompts, generations):
        ret = {}
        for i,p in tqdm(enumerate(prompts)):
            ret[p] = (self.regard.compute(data=generations[i], aggregation='maximum'), generations[i])

        return ret

