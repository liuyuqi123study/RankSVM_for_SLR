from .model.PairwiseLecardPLM import PairwisePLM
from .model.Lawformer import Lawformer
model_list = {
    "pairwise": PairwisePLM,
    'lawformer':Lawformer
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
