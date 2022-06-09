import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from torch_scatter import scatter_mean, scatter_sum
class XLWiCModel(nn.Module):
    def __init__(self, conf) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(conf.model.pretrained_model_name, output_hidden_states=True)
        model_conf = AutoConfig.from_pretrained(conf.model.pretrained_model_name)
        if conf.model.words_combiner == 'cat':
            hidden_size = model_conf.hidden_size * 2
        elif conf.model.words_combiner == 'sum':
            hidden_size = model_conf.hidden_size
        else:
            raise RuntimeError(f'Word Combiner {conf.model.words_combiner} not recognised. Choose between `cat` or `sum`')
        self.linear = nn.Linear(hidden_size, 1, bias=conf.model.use_bias)
        self.criterion = nn.BCEWithLogitsLoss()
        self.activation = nn.Sigmoid()
        self.conf = conf
    
    def forward(self, input_ids, indices_mask, labels=None):
        enc_output = self.encoder(input_ids)
        
        hs_indices = self.conf.model.hidden_states_indices
        hidden_states = [enc_output.hidden_states[x] for x in hs_indices]
        hidden_states = sum(hidden_states)

        if self.conf.model.subword_combiner == 'mean':
            hidden_states_aux = scatter_mean(hidden_states, indices_mask, 1)
        elif self.conf.model.subword_combiner == 'sum':
            hidden_states_aux = scatter_sum(hidden_states, indices_mask, 1)
        else:
            raise RuntimeError(f'Subword Combiner {self.conf.model.subword_combiner} not recognised. Choose between `mean` or `sum`')

        word1_embeddings = hidden_states_aux[:, 1, :]
        word2_embeddings = hidden_states_aux[:, 2, :]

        if self.conf.model.words_combiner == 'cat':
            hidden_states = torch.cat([word1_embeddings, word2_embeddings], -1)
        elif self.conf.model.words_combiner == 'sum':
            hidden_states = word1_embeddings + word2_embeddings
        else:
            raise RuntimeError(f'Word Combiner {self.conf.model.words_combiner} not recognised. Choose between `cat` or `sum`')

        logits = self.linear(hidden_states)
        predictions = self.activation(logits) >= 0.5
        ret = {'logits': logits, 'predictions': predictions}
        if labels is not None:
            loss = self.criterion(logits.squeeze(), labels)
            ret['loss'] = loss
        return ret
