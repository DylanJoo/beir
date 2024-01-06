from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from operator import itemgetter

import os
import datetime
import logging
import argparse 

from utils import load_corpus, load_results, load_pseudo_queries
from utils import LoggingHandler

from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--pseudo_queries", type=str, default=None)
    # 
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #### Reranking using Cross-Encoder model
    reranker = CrossEncoder(args.model_name, num_labels=1)

    #### Load data
    corpus_texts = load_corpus(os.path.join(args.dataset, 'corpus.jsonl'))
    pseudo_queries = load_pseudo_queries(args.pseudo_queries)

    #### Prepare examples
    train_samples = []
    for docid in pseudo_queries:
        document = corpus_texts[docid]
        for query, score in pseudo_queries[docid]:
            #### [SETTING I]: One positive and one negative
            if score == 1 or score == 0:
                train_samples.append(InputExample(texts=[query, document], label=score))
                train_samples.append(InputExample(texts=[document, query], label=score))

    #### Prepare dataloader
    train_dataloader = DataLoader(
            train_samples, 
            batch_size=args.batch_size,
            collate_fn=reranker.smart_batching_collate_text_only,
            num_workers=0,
            shuffle=True, 
    )

    #### Saving benchmark times
    start = datetime.datetime.now()

    #### Start training
    reranker.fit(
            train_dataloader=train_dataloader,
            loss_fct=None,
            epochs=args.num_epochs,
            warmup_steps=0,
            output_path=args.output_path
    )
# cross encoder params
# train_dataloader: DataLoader,
# evaluator: SentenceEvaluator = None,
# epochs: int = 1,
# loss_fct = None,
# activation_fct = nn.Identity(),
# scheduler: str = 'WarmupLinear',
# warmup_steps: int = 10000,
# optimizer_class: Type[Optimizer] = torch.optim.AdamW,
# optimizer_params: Dict[str, object] = {'lr': 2e-5},
# weight_decay: float = 0.01,
# evaluation_steps: int = 0,
# output_path: str = None,
# save_best_model: bool = True,
# max_grad_norm: float = 1,
# use_amp: bool = False,
# callback: Callable[[float, int, int], None] = None,
# show_progress_bar: bool = True

    #### Measure time to 
    end = datetime.datetime.now()
    
    #### Measuring time taken in ms (milliseconds)
    time_taken = (end - start)
    time_taken = time_taken.total_seconds() * 1000
    logging.info("Training time: {:.2f}ms".format(time_taken))
