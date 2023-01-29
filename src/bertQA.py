from typing import Tuple

import torch
from torch import nn
from transformers import BertModel, AutoConfig

from config import BertQAConfig


class QAModel(nn.Module):
    """Finetuned to QA BERT."""

    def __init__(self) -> None:
        """Define model architecture."""
        super(QAModel, self).__init__()
        cfg = AutoConfig.from_pretrained(BertQAConfig().artifacts_dir)
        self.bert = BertModel(cfg)
        self.drop_out = nn.Dropout(0.1)
        self.l1 = nn.Linear(768 * 2, 2)
        self.linear_stack = nn.Sequential(
            self.drop_out,
            self.l1,
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param input_ids: tensor with input tokens
        :param attention_mask: tensor  with attention mask
        :param token_type_ids:  tensor with types of tokens (question or answer)
        """
        model_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states = model_output[2]
        out = torch.cat((hidden_states[-1], hidden_states[-3]), dim=-1)
        logits = self.linear_stack(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
