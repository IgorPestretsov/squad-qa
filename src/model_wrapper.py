from typing import Tuple

import torch
from transformers import AutoTokenizer, BertTokenizerFast

from bertQA import QAModel
from config import BertQAConfig
from data_containers import PredictRequestSchema


class Pipeline:
    """Pipeline for QA on squad v2 dataset."""

    def __init__(self) -> None:
        """Initialize pipeline with the config, model and tokenizer."""
        self.config = BertQAConfig()
        self.model, self.tokenizer = self.load_artifacts()

    def predict(self, input: PredictRequestSchema) -> str:
        """
        Predict answer from context.

        :param input: context and question
        :return: answer
        """
        inputs = self.tokenizer.encode_plus(input.question, input.context, return_tensors='pt').to('cpu')
        with torch.no_grad():
            output_start, output_end = self.model(**inputs)

            answer_start = int(torch.argmax(output_start))
            answer_end = int(torch.argmax(output_end))

            if answer_end != 0:
                answer_end += 1

            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
            return answer

    def load_artifacts(self) -> Tuple[QAModel, BertTokenizerFast]:
        """
        Load pretrained model and tokenizer.

        :return: Tuple[QAModel, BertTokenizerFast
        """
        model = QAModel()
        tokenizer = AutoTokenizer.from_pretrained(self.config.artifacts_dir)
        model.load_state_dict(torch.load(self.config.trained_model_path))
        model.eval()
        return model, tokenizer
