import torch
from typing import Dict, Type, Callable, List, Tuple
from sentence_transformers.cross_encoder import CrossEncoder

class PACECrossEncoder(CrossEncoder):

    def __init__(self, 
        model_name: str, 
        num_labels: int = None, 
        max_length: int = None, 
        device: str = None, 
        tokenizer_args: Dict = {},
        automodel_args: Dict = {}, 
        default_activation_function = None, 
        document_centric: bool = True
    ):
        super().__init__(model_name, num_labels, max_length, 
                device, tokenizer_args, automodel_args, 
                default_activation_function
        )
        self.document_centric = document_centric

    def smart_batching_collate(self, batch):
        texts = [[], []]
        labels = []

        for example in batch:
            center = example.center
            for i, text in enumerate(example.texts):
                if self.document_centric:
                    texts[0].append(text.strip()) # different queries
                    texts[1].append(center.strip())
                    labels.append(example.labels[i])
                else: # the standard query-centric training
                    texts[0].append(center.strip()) 
                    texts[1].append(text.strip()) 
                    labels.append(example.label[i])

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels
