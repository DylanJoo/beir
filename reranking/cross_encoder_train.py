# from sentence_transformers.cross_encoder import CrossEncoder
from cross_encoder import PACECrossEncoder
from sentence_transformers import InputExample
from operator import itemgetter

import os
import json
import datetime
import logging
import argparse 

from torch.utils.data import DataLoader

from pacerr.filters import filter_function_map
from pacerr.utils import load_corpus, load_results, load_pseudo_queries
from pacerr.utils import LoggingHandler
from pacerr.inputs import GroupInputExample
from pacerr.losses import PairwiseHingeLoss

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
    # training
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--bidirectional", action='store_true', default=False)
    parser.add_argument("--pointwise", action='store_true', default=False)
    parser.add_argument("--groupwise", action='store_true', default=False)
    parser.add_argument("--document_centric", action='store_true', default=False)
    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #### Reranking using Cross-Encoder model
    reranker = PACECrossEncoder(args.model_name, 
                                num_labels=1, 
                                document_centric=args.document_centric)

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

        if args.pointwise:
            for query, score in pairs:
                train_samples.append(InputExample(texts=[query, document], label=score))
                # if args.bidirectional:
                #     train_samples.append(PointInputExample(texts=[document, query], label=score))

        if args.groupwise:
            queries, scores = map(list, (list(zip(*pairs))) )
            train_samples.append(GroupInputExample(
                center=document, texts=queries, labels=scores
            ))

    #### Prepare dataloader
    n = 1 
    n = len(scores) if args.groupwise else n
    train_dataloader = DataLoader(
            train_samples, 
            batch_size=args.batch_size // n,
            collate_fn=reranker.smart_batching_collate, # in fact, no affect
            num_workers=0,
            shuffle=True, 
    )


    #### Prepare losses
    if args.pointwise:
        loss_fct = None

    if args.groupwise:
        loss_fct = PairwiseHingeLoss(
                example_per_group=n, 
                batch_size=args.batch_size,
                margin=0, 
                reduction='mean'
        )

    #### Saving benchmark times
    start = datetime.datetime.now()

    #### Start training
    reranker.fit(
            train_dataloader=train_dataloader,
            loss_fct=loss_fct,
            epochs=args.num_epochs,
            warmup_steps=len(train_dataloader),
            optimizer_params={'lr': args.learning_rate},
            output_path=args.output_path # only save when evaluation
    )
    reranker.save(args.output_path)

    #### Measure time to 
    end = datetime.datetime.now()
    
    #### Measuring time taken in ms (milliseconds)
    time_taken = (end - start)
    time_taken = time_taken.total_seconds() * 1000
    logging.info("Training time: {:.2f}ms".format(time_taken))
