import torch
from pacerr.inputs import GroupInputExample
from cross_encoder import StandardCrossEncoder

class PACECrossEncoder(StandardCrossEncoder):

    def setup_tunable(self, allowed):
        for name, param in self.model.named_parameters():
            if allowed in name:
                param.requires_grad = True
                print(f"{name} is tunable; the others are freezed.")
            else:
                param.requires_grad = False

    def smart_batching_collate(self, batch):
        """Recast all the input into query-centric""" 
        tokenized = labels = None
        batch_for_qc = _reverse_batch_negative(batch)
        (texts_0, texts_1), scores = self.collate_from_inputs(batch_for_qc, batch)

        tokenized = self.tokenizer(texts_0, texts_1, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        tokenized.to(self._target_device)
        labels = torch.tensor(scores, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)
        return None, None, tokenized, labels

    def collate_from_inputs(self, batch_qc, batch_dc):
        assert len(batch_qc) == len(batch_dc), 'inconsistent size'

        sent_left = []
        sent_right = []
        labels = []

        for example_qc, example_dc in zip(batch_qc, batch_dc):
            query_1 = example_qc.center.strip()
            document_1 = example_dc.center.strip()
            sent_left.append(query_1)
            sent_right.append(document_1)
            labels.append(1)

            for i, document in enumerate(example_qc.texts[1:]):
                # query_is_center:
                sent_left.append(query_1) 
                sent_right.append(document.strip())
                labels.append(0)

            for i, query in enumerate(example_dc.texts[1:]):
                # document_is_center:
                if self.change_dc_to_qq:
                    sent_left.append(query_1) 
                    sent_right.append(query.strip())
                else:
                    sent_left.append(query.strip()) 
                    sent_right.append(document_1)
                labels.append(0)

        return (sent_left, sent_right), labels


def _reverse_batch_negative(batch):
    batch_return = []

    centers = [ex.center.strip() for ex in batch]
    batch_sides = [ex.texts for ex in batch] 
    batch_labels = [ex.labels for ex in batch] 

    for i, (sides, labels) in enumerate(zip(batch_sides, batch_labels)):
        positive = [centers[i]]
        ibnegatives = centers[:i] + centers[(i+1):]

        for j, (side, label) in enumerate(zip(sides, labels)):
            # [NOTE] So far, we use only the first one.
            if j == 0:
                batch_return.append(GroupInputExample(
                    center=side, 
                    texts=positive+ibnegatives,
                    labels=[1]+[0]*len(ibnegatives)
                ))
    return batch_return
