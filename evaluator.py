from genbit.genbit_metrics import GenBitMetrics
import csv
import evaluate

class AutoEvalEvaluator:
    def __init__(self):
        pass

    @staticmethod
    def load_prompts_csv(prompt_fp):
        with open(prompt_fp, "r") as f:
            ret = [p[-1] for p in csv.reader(f)]

    def evaluate():
        raise NotImplementedError()


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
        self.genbit = GenBitWrapper()

    def evaluate(self, targetLLM, num_gen=1):
        for p in self.prompts:
            for i in range(num_gen):
                self.genbit(targetLLM.generate(p))

        return self.genbit.evaluate()


class HonestEvaluator(AutoEvalEvaluator):
    def __init__(self, prompts_fp):
        self.prompts = AutoEvalEvaluator.load_prompts_csv(prompts_fp)
        self.honest = evaluate.load("honest", "en")

    def evaluate(self, targetLLM, num_gen=1):
        for p in self.prompts:
            for i in range(num_gen):
                generations.append(targetLLM.generate(p))
        return self.honest.compute(predictions=generations)



