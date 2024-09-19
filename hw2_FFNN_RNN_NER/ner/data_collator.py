# AUTO-GENERATED (DO NOT MODIFY)
# NET IDS: KL873

import logging
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Any

import numpy as np
import torch

from ner.data_processing.constants import NER_ENCODING_MAP, PAD_NER_TAG
from ner.data_processing.tokenizer import Tokenizer


class DataCollator(object):
    def __init__(
        self,
        tokenizer: Tokenizer,
        padding: Union[str, bool] = "longest",
        max_length: Optional[int] = None,
        padding_side: str = "right",
        truncation_side: str = "right",
        pad_tag: str = PAD_NER_TAG,
        text_colname: str = "text",
        label_colname: str = "NER",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.pad_tag = pad_tag
        self.text_colname = text_colname
        self.label_colname = label_colname

    def _get_max_length(self, data_instances: List[Dict[str, Any]]) -> Optional[int]:
        if not ((self.padding == "longest" or self.padding) and self.max_length is None):
            logging.warning(
                f"both max_length={self.max_length} and padding={self.padding} provided; ignoring "
                f"padding={self.padding} and using max_length={self.max_length}"
            )
            self.padding = "max_length"

        if self.padding == "longest" or (isinstance(self.padding, bool) and self.padding):
            return max([len(data_instance[self.text_colname]) for data_instance in data_instances])
        elif self.padding == "max_length":
            return self.max_length
        elif isinstance(self.padding, bool) and not self.padding:
            return None
        raise ValueError(f"padding strategy {self.padding} is invalid")

    @staticmethod
    def _process_labels(labels: List) -> torch.Tensor:
        return torch.LongTensor([NER_ENCODING_MAP[label] for label in labels])

    def __call__(self, data_instances: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Documentation: https://pages.github.coecis.cornell.edu/cs4740/hw2-fa23/ner.data_processing.data_collator.html.
        """
        # TODO-1.2-1
        batch_size = len(data_instances)
        batch_max_length = self._get_max_length(data_instances)

        input_ids = torch.empty((batch_size, batch_max_length),dtype=torch.long)
        padding_mask = torch.empty((batch_size, batch_max_length), dtype=torch.long)
        
        label_list = []
        for i, sample in enumerate(data_instances):
          text = sample[self.text_colname]
          tokenized_data = self.tokenizer.tokenize(input_seq=text, max_length=batch_max_length, padding_side=self.padding_side, truncation_side=self.truncation_side)
          tokenized_input_ids = tokenized_data['input_ids']
          tokenized_padding_mask = tokenized_data['padding_mask']
          input_ids[i, :len(tokenized_input_ids)] = torch.LongTensor(tokenized_input_ids)
          padding_mask[i, :len(tokenized_padding_mask)] = torch.Tensor(tokenized_padding_mask)
          padding_mask = torch.squeeze(padding_mask)

          max_length = len(tokenized_input_ids)
          if self.label_colname in sample.keys():
            labels = sample[self.label_colname]
            num_of_padded = max_length-len(labels)
            if self.padding_side == 'right':
              labels = labels + num_of_padded * [self.pad_tag]
            else:
              labels = num_of_padded * [self.pad_tag] + labels
            labels_batch_tensor = self._process_labels(labels)
            label_list.append(labels_batch_tensor)

        # If labels are present, pad them to have equal length before stacking
        if label_list:
            max_label_length = max(len(label) for label in label_list)
            padded_labels = [torch.cat([label, torch.zeros(max_label_length - len(label), dtype=label.dtype)]) for label in label_list]
            labels_tensor = torch.stack(padded_labels, dim=0)
            return {"input_ids": input_ids, "padding_mask": padding_mask, "labels": labels_tensor}
        else:
            return {"input_ids": input_ids, "padding_mask": padding_mask}
        
        
        
