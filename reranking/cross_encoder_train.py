from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from cross_encoder import StandardCrossEncoder, PACECrossEncoder
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
from pacerr.utils import load_queries, load_and_convert_qrels
from pacerr.utils import LoggingHandler
from pacerr.inputs import GroupInputExample
from pacerr.losses import PointwiseMSELoss
from pacerr.losses import PairwiseHingeLoss, GroupwiseHingeLoss, GroupwiseHingeLossV1
from pacerr.losses import CELoss, GroupwiseCELossV1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--pseudo_queries", type=str, default=None)
    parser.add_argument("--qrels", type=str, default=None)
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
    parser.add_argument("--margin", type=int, default=1)
    # evaluation
    parser.add_argument("--do_eval", action='store_true', default=False)
    # saving 
    parser.add_argument("--save_last", action='store_true', default=False)
    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    #### Reranking using Cross-Encoder model
    if 'pointwise' in args.objective:
        reranker = StandardCrossEncoder(args.model_name, num_labels=1,)
    else:
        reranker = PACECrossEncoder(args.model_name, 
                                    num_labels=1, 
                                    document_centric=args.document_centric)

    #### Add wandb 
    wandb.init(
            name=f"{args.pseudo_queries.split('/')[-1]} @ {args.objective}",
            config=reranker.config
    )
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
    train_dataloader = DataLoader(
            train_samples, 
            batch_size=args.batch_size,
            collate_fn=reranker.smart_batching_collate, # in fact, no affect
            num_workers=0,
            shuffle=True, 
    )
    n = 1 if 'pointwise' in args.objective else len(scores)


    #### Prepare losses
    # Hinge
    if 'pairwise_hinge' in args.objective:
        logging.info("Using objective: PairwiseHingeLoss")
        loss_fct = PairwiseHingeLoss(
                examples_per_group=n, 
                margin=args.margin, 
                reduction='mean'
        )
    if 'groupwise_hinge' in args.objective:
        logging.info("Using objective: GroupwiseHingeLoss")
        loss_fct = GroupwiseHingeLoss(
                examples_per_group=n, 
                margin=args.margin,
                reduction='mean'
        )
    if 'groupwise_hinge_v1' in args.objective:
        logging.info("Using objective: GroupwiseHingeLossV1")
        loss_fct = GroupwiseHingeLossV1(
                examples_per_group=n, 
                margin=args.margin,
                stride=1, 
                dilation=1,
                reduction='mean'
        )

    # CE
    if 'pairwise_ce' in args.objective:
        logging.info("Using objective: PairwiseCELoss")
        assert n == 2, 'the filtering function can only output 2 only'
        loss_fct = CELoss(
                examples_per_group=2, # a positive and a negative
                reduction='mean'
        )
    if 'groupwise_ce' in args.objective:
        logging.info("Using objective: GroupwiseCELoss")
        assert n > 2, 'the filtering function can only output larger than 2'
        loss_fct = CELoss(
                examples_per_group=n, # a positive and multiple negatives
                reduction='mean'
        )
    if 'groupwise_ce_v1' in args.objective:
        logging.info("Using objective: GroupwiseCELossV1")
        loss_fct = GroupwiseCELossV1(
                examples_per_group=n, 
                margin=args.margin,
                stride=1, 
                dilation=1,
                reduction='mean'
        )

    if 'pointwise_mse' in args.objective:
        loss_fct = MSELoss(reduction='mean')
        logging.info("Using objective: PointwiseMSELoss")

    if 'pointwise_bce' in args.objective:
        loss_fct = None # default in sentence bert
        logging.info("Using objective: BCELogitsLoss")

    if 'distillation_mse' in args.objective:
        loss_fct = MSELoss(reduction='mean')
        logging.info("Using objective: DistillationMSELoss")

    #### Saving benchmark times
    start = datetime.datetime.now()

    #### Add evaluation
    ##### Load Evaluation data with the format: [{'query': '', 'positive': [], 'negative': []}, ...]
    if args.do_eval:
        queries = load_queries(os.path.join(args.dataset, 'queries.jsonl'))
        dev_samples = load_and_convert_qrels(
                path=args.qrels,
                queries=queries,
                corpus_texts=corpus_texts
        )
        evaluator = CERerankingEvaluator(dev_samples, name='test')

    #### Start training
    logging.info(f"The dataset has {len(train_dataloader)} batch")
    reranker.fit(
            train_dataloader=train_dataloader,
            loss_fct=loss_fct,
            evaluator=evaluator,
            epochs=args.num_epochs,
            evaluation_steps=len(train_dataloader) // 5,  
            warmup_steps=len(train_dataloader) // 10,
            optimizer_params={'lr': args.learning_rate},
            output_path=args.output_path, # only save when evaluation
            wandb=wandb
    )
    if args.save_last:
        reranker.save(args.output_path)

    #### Measure time to 
    end = datetime.datetime.now()
    
    #### Measuring time taken in ms (milliseconds)
    time_taken = (end - start)
    time_taken = time_taken.total_seconds() * 1000
    logging.info("Training time: {:.2f}ms".format(time_taken))
