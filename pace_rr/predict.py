from sentence_transformers.cross_encoder import CrossEncoder
from operator import itemgetter

import os
import datetime
import logging

from utils import load_queries, load_corpus, load_results
from utils import LoggingHandler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--input_run", type=str, default=None)
    parser.add_argument("--output_run", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    # 
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])


    #### Reranking using Cross-Encoder model
    reranker = CrossEncoder(args.model_name)

    #### Load data
    queries = load_queries(os.path.join(args.dataset, 'queries.jsonl'))
    corpus = load_corpus(os.path.join(args.dataset, 'corpus.jsonl'))
    results = load_results(args.input_run, topk=args.top_k)

    #### Saving benchmark times
    time_taken_all = {}

    for query_id in queries:
        query = queries[query_id]
        result = results[query_id]
        
        #### Measure time to retrieve top-100 BM25 documents using single query latency
        start = datetime.datetime.now()
        
        #### Measure time to rerank top-100 BM25 documents using CE
        sentence_pairs = [[query, corpus_texts[docid]] for docid in result]
        scores = reranker.predict(sentence_pairs, batch_size=args.batch_size, show_progress_bar=False)
        hits = {result[idx]: scores[idx] for idx in range(len(scores))}            
        sorted_results = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)} 
        end = datetime.datetime.now()
        
        #### Measuring time taken in ms (milliseconds)
        time_taken = (end - start)
        time_taken = time_taken.total_seconds() * 1000
        time_taken_all[query_id] = time_taken
        logging.info("{}: {} {:.2f}ms".format(query_id, query, time_taken))

    time_taken = list(time_taken_all.values())
    logging.info("Average time taken: {:.2f}ms".format(sum(time_taken)/len(time_taken_all)))
