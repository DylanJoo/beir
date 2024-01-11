from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from cross_encoder import PACECrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import InputExample
from operator import itemgetter
import random

import os
import json
import datetime
import logging
import argparse 
import wandb

from torch.utils.data import DataLoader

from pacerr.filters import filter_function_map
from pacerr.utils import load_corpus, load_results, load_pseudo_queries
from pacerr.utils import LoggingHandler
from pacerr.inputs import GroupInputExample
from pacerr.losses import PairwiseHingeLoss, PairwiseLCELoss
from pacerr.losses import CombinedLoss
from pacerr.losses import GroupwiseHingeLoss, GroupwiseLCELoss

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
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--document_centric", action='store_true', default=False)
    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #### Reranking using Cross-Encoder model
    if 'pointwise' in args.objective:
        reranker = CrossEncoder(args.model_name, num_labels=1,)
    else:
        reranker = PACECrossEncoder(args.model_name, 
                                    num_labels=1, 
                                    document_centric=args.document_centric)

    #### Add wandb 
    wandb.init()
    wandb.watch(reranker.model, log_freq=10)

    #### Load data
    corpus_texts = load_corpus(os.path.join(args.dataset, 'corpus.jsonl'))
    pseudo_queries = load_pseudo_queries(args.pseudo_queries)

    #### Prepare a filter
    filter_args = json.loads(args.filtering)
    filter_name = filter_args.pop('name', 'testing')
    filter_fn = filter_function_map[filter_name]

    #### Prepare examples
    train_samples = []
    dev_samples = []
    for docid in pseudo_queries:
        document = corpus_texts[docid]

        #### Filtering
        pairs = filter_fn(pseudo_queries[docid], **filter_args)

        if 'pointwise' in args.objective:
            for query, score in pairs:
                train_samples.append(InputExample(texts=[query, document], label=score))
        else:
            queries, scores = map(list, (list(zip(*pairs))) )
            train_samples.append(GroupInputExample(
                center=document, texts=queries, labels=scores
            ))

    #### Prepare dataloader
    n = 1 if 'pointwise' in args.objective else len(scores)
    train_dataloader = DataLoader(
            train_samples, 
            batch_size=args.batch_size // n,
            collate_fn=reranker.smart_batching_collate, # in fact, no affect
            num_workers=0,
            shuffle=True, 
    )


    #### Prepare losses
    if 'pairwise_hinge' in args.objective:
        logging.info("Using objective: PairwiseHingeLoss")
        loss_fct = PairwiseHingeLoss(
                examples_per_group=n, 
                margin=1, 
                reduction='mean'
        )
    if 'pairwise_lce' in args.objective:
        logging.info("Using objective: LCELoss")
        loss_fct = PairwiseLCELoss(
                examples_per_group=n, reduction='mean'
        )
    if 'groupwise_hinge' in args.objective:
        logging.info("Using objective: GroupwiseHingeLoss")
        loss_fct = GroupwiseHingeLoss(
                examples_per_group=n, reduction='mean'
        )
    if 'groupwise_lce' in args.objective:
        logging.info("Using objective: GroupwiseLCELoss")
        loss_fct = GroupwiseLCELoss(
                examples_per_group=n, reduction='mean'
        )

    if 'combined_v1' in args.objective:
        logging.info("Using objective: BCELogitsLoss + PairwiseHingeLoss")
        loss_fct = CombinedLoss(
                add_hinge_loss=True,
                examples_per_group=n, 
                reduction='mean'
        )
    if 'combined_v2' in args.objective:
        logging.info("Using objective: BCELogitsLoss + LCELoss")
        loss_fct = CombinedLoss(
                add_lce_loss=True,
                examples_per_group=n, 
                reduction='mean'
        )
    if 'pointwise_mse' in args.objective:
        loss_fct = PointwiseMSELoss()
        logging.info("Using objective: MSELoss")

    if 'pointwise_bce' in args.objective:
        loss_fct = None # default in sentence bert
        logging.info("Using objective: BCELogitsLoss")

    #### Saving benchmark times
    start = datetime.datetime.now()

    #### Add evaluation
    evaluator = None
    # evaluator = CEBinaryClassificationEvaluator(
    #         dev_samples, name='train-eval'
    # )

    #### Start training
    logging.info(f"The dataset has {len(train_dataloader)} batch")
    reranker.fit(
            train_dataloader=train_dataloader,
            loss_fct=loss_fct,
            evaluator=evaluator,
            epochs=args.num_epochs,
            warmup_steps=len(train_dataloader) // 10,
            optimizer_params={'lr': args.learning_rate},
            output_path=args.output_path, # only save when evaluation
            wandb=wandb
    )
    reranker.save(args.output_path)

    #### Measure time to 
    end = datetime.datetime.now()
    
    #### Measuring time taken in ms (milliseconds)
    time_taken = (end - start)
    time_taken = time_taken.total_seconds() * 1000
    logging.info("Training time: {:.2f}ms".format(time_taken))
