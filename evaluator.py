from genbit.genbit_metrics import GenBitMetrics

class GenBitEvaluator():
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