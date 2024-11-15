import logging
import os
import torch
from torch.autograd import Variable
from timeit import default_timer as timer

from tools.eval_tool import gen_time_str, output_value
logger = logging.getLogger(__name__)
import json


def test(parameters, config, gpu_list):
    output_function=parameters['output_function']
    model = parameters["model"]
    dataset = parameters["test_dataset"]
    model.eval()

    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    #output_function = parameters["output_function"]
    output_info = "testing"

    output_time = config.getint("output", "output_time")
    step = -1
    result = []


    res_scores=[]
    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])
                if torch.backends.mps.is_available():
                    data[key]=Variable(data[key].to('mps'))

        results = model(data, config, gpu_list, acc_result, "test")
        loss, acc_result = results["loss"], results["acc_result"]
        total_loss += float(loss)
        cnt += 1

        res_scores += list(zip(results["index"], results["score"]))
        if step % output_time == 0:
            delta_t = timer() - start_time
            #output_info = output_function(acc_result, config)

            output_value(0, "test", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, '\r', config)

    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError
    
    
    predictions = {}
    
    for res in res_scores:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        if res[0][0] not in predictions:
            predictions[res[0][0]] = []
        predictions[res[0][0]].append((res[0][1], res[1]))

    for key in predictions:
        predictions[key].sort(key = lambda x:x[1], reverse = True)
        predictions[key] = [int(res[0]) for res in predictions[key]]

    os.makedirs(config.get("data", "result_path"), exist_ok=True)
    fout = open(os.path.join(config.get("data", "result_path"), "prediction_epoch_0_long.json"), "w")
    print(json.dumps(predictions), file = fout)

    fout.close()
    delta_t = timer() - start_time
    output_info = output_function(acc_result, config)
    output_value(0, "test", "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                 "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    return result