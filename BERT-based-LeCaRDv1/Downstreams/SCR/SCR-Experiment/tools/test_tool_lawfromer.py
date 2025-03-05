import logging
import os
import torch
from torch.autograd import Variable
from timeit import default_timer as timer
import csv

from tools.eval_tool import gen_time_str, output_value

logger = logging.getLogger(__name__)
from transformers import AutoModel, AutoTokenizer

def test(path):
    dataset=
    tokenizer=AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = AutoModel.from_pretrained("thunlp/Lawformer")