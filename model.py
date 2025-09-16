from torch import nn

class TransformerClassifier(nn.Module):
    def __init__(self, plm_config, labels_to_ids):
        super().__init__()
        self.transformer_block = nn.Transformer(d_model=plm_config.hidden_size,
                                                nhead=plm_config.num_attention_heads,
                                                dim_feedforward=plm_config.intermediate_size,
                                                dropout=plm_config.hidden_dropout_prob,
                                                activation=plm_config.hidden_act,
                                                layer_norm_eps=plm_config.layer_norm_eps,
                                                batch_first=True,
                                                norm_first=True)
        self.cls_layer = nn.Linear(plm_config.hidden_size, len(labels_to_ids))
    def forward(self, prompts_contexts_logit, responses_logit):
        transformer_logit = self.transformer_block(src=prompts_contexts_logit, tgt=responses_logit)
        output_logit = self.cls_layer(transformer_logit)
        return output_logit