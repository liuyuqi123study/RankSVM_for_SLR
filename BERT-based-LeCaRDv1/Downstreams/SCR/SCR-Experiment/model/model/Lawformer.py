from transformers import AutoModelForMaskedLM,AutoModelForPreTraining,LongformerConfig,LongformerForMaskedLM,AutoModel,AutoConfig
import torch
from torch import nn

class Lawformer(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Lawformer, self).__init__()
        plm_path = config.get('train', 'PLM_path')
        self.encoder=AutoModel.from_pretrained(plm_path)
    
        self.plm_config = AutoConfig.from_pretrained(plm_path)
        self.hidden_size=self.plm_config.hidden_size
        self.fc=nn.Linear(self.hidden_size,2)
        self.encoder = nn.DataParallel(self.encoder, device_ids=[0])
        self.fc = nn.DataParallel(self.fc, device_ids=[0])
        self.criterion = nn.CrossEntropyLoss()
    def init_multi_gpu(self, device, config, *args, **params):
      
        print('init multi gpus')

    def forward(self, data, config, gpu_list, acc_result, mode):
        pair=1
        batch,seq_len=data['inputx'].shape[0],data['inputx'].shape[2]
        inputx = data["inputx"].view(batch * pair, seq_len)
        mask = data["mask"].view(batch * pair, seq_len)
        segment = data["segment"].view(batch * pair, seq_len)

        ret = self.encoder(input_ids=inputx, attention_mask=mask,token_type_ids=segment)
        y = ret['pooler_output'].squeeze(1)
        result = self.fc(y)
        loss = self.criterion(result, data["labels"])
        acc_result = accuracy(result, data["labels"], acc_result)

        if mode == "train":
            return {"loss": loss, "acc_result": acc_result}
        else:
            score = torch.softmax(result, dim=1)  # batch, 2
            return {"loss": loss, "acc_result": acc_result, "score": score[:, 1].tolist(), "index": data["index"]}
def accuracy(logit, label, acc_result):
        if acc_result is None:
            acc_result = {'right': 0, 'actual_num': 0, 'pre_num': 0}
        pred = torch.max(logit, dim=1)[1]
        acc_result['pre_num'] += int((pred == 1).sum())
        acc_result['actual_num'] += int((label == 1).shape[0])
        acc_result['right'] += int((pred[label == 1] == 1).sum())
        return acc_result