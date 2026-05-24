from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig, XLMRobertaModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput


@dataclass
class QATCModelOutput(QuestionAnsweringModelOutput):
    rational_tag_logits: Optional[torch.FloatTensor] = None


class QATCConfig(PretrainedConfig):
    model_type = "qatc"

    def __init__(
        self,
        model_name: Optional[str] = None,
        freeze_text_encoder: bool = False,
        alpha: float = 1.0,
        beta: float = 0.01,
        lambda_sparse: float = 0.01,
        lambda_continuity: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name or "microsoft/infoxlm-large"
        self.freeze_text_encoder = freeze_text_encoder
        self.alpha = alpha
        self.beta = beta
        self.lambda_sparse = lambda_sparse
        self.lambda_continuity = lambda_continuity


class RationalTaggingHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.W1(hidden_states)
        logits = F.relu(logits)
        return torch.sigmoid(self.w2(logits))


class QATCForQuestionAnswering(PreTrainedModel):
    config_class = QATCConfig
    base_model_prefix = "qa_model"

    def __init__(self, config: QATCConfig):
        super().__init__(config)
        self.config = config

        base_model_source = self.config.model_name
        local_base_model = os.path.isdir(str(base_model_source or ""))
        base_config = AutoConfig.from_pretrained(
            base_model_source,
            trust_remote_code=True,
            local_files_only=local_base_model,
        )
        self.model = XLMRobertaModel(base_config)

        if getattr(self.config, "freeze_text_encoder", False):
            for param in self.model.parameters():
                param.requires_grad = False

        hidden_size = int(self.model.config.hidden_size)
        self.qa_outputs = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        )
        self.tagging = RationalTaggingHead(hidden_size)
        if hasattr(self.model, "pooler"):
            self.model.pooler = None
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_positions=None,
        end_positions=None,
        tagging_labels=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        sequence_output = outputs[0]

        rational_tag_logits = self.tagging(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if (
            start_positions is not None
            and end_positions is not None
            and tagging_labels is not None
        ):
            raise NotImplementedError(
                "Training loss is not implemented in the standalone inference wrapper."
            )

        if not return_dict:
            return (total_loss, start_logits, end_logits, rational_tag_logits) + outputs[2:]

        return QATCModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rational_tag_logits=rational_tag_logits,
        )
