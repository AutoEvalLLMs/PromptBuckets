# Generator.py is a file that contains the generator class and any helper functions necessary:
import transformers
from transformers import GenerationConfig, GenerationMixin
import torch
from collections import OrderedDict, defaultdict

DEFAULT_TARGET_GENERATOR_CONFIG = {
    "api_model" : False,
    "max_new_tokens" : 100, # number of tokens to generate outside of the prompt
    "num_generations" : 5, # number of generations per prompt
}

class TargetGeneratorConfig(GenerationConfig):
    """
    SUPERCLASS for TargetGeneratorConfig
    api_model: False IF using a weights-based model. Otherwise one of {'gpt', 'claude'}
    
    This class handles the generation strategy given parameters as follows:

    greedy decoding by calling greedy_search() if num_beams=1 and do_sample=False
    contrastive search by calling contrastive_search() if penalty_alpha>0. and top_k>1
    multinomial sampling by calling sample() if num_beams=1 and do_sample=True
    beam-search decoding by calling beam_search() if num_beams>1 and do_sample=False
    beam-search multinomial sampling by calling beam_sample() if num_beams>1 and do_sample=True
    diverse beam-search decoding by calling group_beam_search(), if num_beams>1 and num_beam_groups>1
    constrained beam-search decoding by calling constrained_beam_search(), if constraints!=None or force_words_ids!=None
    assisted decoding by calling assisted_decoding(), if assistant_model is passed to .generate()
    You do not need to call any of the above methods directly. Pass custom parameter values to ‘.generate()‘. To learn more about decoding strategies refer to the text generation strategies guide.

    """
    def __init__(self, api_model: str = False, **kwargs):
        super().__init__(GenerationConfig, **kwargs)
        self.api_model = api_model 
        self.num_generations = kwargs.get('num_generations', 5)
        
class TargetGenerator(GenerationMixin):
    """
    Class inheriting from GenerationMixin updating methods to support api_models AND handle many generations / prompt
    """
    def __init__(self) -> None:
        super().__init__()

    def  generate_api(self, input_text, **kwargs):
        """
        Generate text using the API model
        """
        raise NotImplementedError
    
    def generate_from_prompt(self, prompt, **kwargs):
        """
        Generate text from a prompt
        """
        raise NotImplementedError
    
