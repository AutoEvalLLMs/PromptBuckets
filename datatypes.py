class PromptSet():
    def __init__(self) -> None:
        self.data = 0
        self.instructions = list(self.data.keys())

    def _unroll_prompts(self):
        """
        creates a list of tuples with each element being:
            (likert-scale-value, prompt)
        Unrolled with regards to instruction
        """
        self.unrolled_prompts = []
        for i in self.instructions:
            for k,v in self.data[i].items:
                self.unrolled_prompts.append((k,v))

    def from_dict(self, d):
        self.data = d

    def __iter__(self):
        for i in range(len(self.ps)):
            yield self.ps[i]
