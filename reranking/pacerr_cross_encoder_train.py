from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
from operator import itemgetter

import os
import json
import datetime
import logging
import argparse 
from pacerr_filter import filter_function_map

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
    # setting
    parser.add_argument("--filtering", type=str, default="{}")
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

    #### Prepare a filter
    filter_args = json.loads(args.filtering)
    filter_name = filter_args.pop('name', 'testing')
    filter_fn = filter_function_map[filter_name]

    #### Prepare examples
    train_samples = []
    for docid in pseudo_queries:
        document = corpus_texts[docid]

        #### Filtering
        pairs = filter_fn(pseudo_queries[docid], **filter_args)
        for query, score in pairs:
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
            output_path=args.output_path # only save when evaluation
    )
    reranker.save(args.output_path)

    #### Measure time to 
    end = datetime.datetime.now()
    
    #### Measuring time taken in ms (milliseconds)
    time_taken = (end - start)
    time_taken = time_taken.total_seconds() * 1000
    logging.info("Training time: {:.2f}ms".format(time_taken))
