import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import collections
from uilts import load_corpus, load_queries, load_input_run_with_topk
from utils import batch_iterator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--input_run", type=str, default=None)
    parser.add_argument("--output_run", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    # 
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    # load data
    queries = load_queries(os.path.join(args.dataset, 'queries.jsonl'))
    corpus = load_corpus(os.path.join(args.dataset, 'corpus.jsonl'))
    pairs_in, pairs_out = load_input_run_with_topk(args.input_run, args.topk)

    # prepare dataset
    features = tokenizer(
            ['How many people live in Berlin?', 'How many people live in Berlin?'], 
            ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', \
                'New York City is famous for the Metropolitan Museum of Art.'],  
            padding=True, truncation=True, return_tensors="pt"
    )

    # miniLM
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    model.eval()
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # predict
    pairs_new = []
    iterator = batch_iterator(pairs_in, args.batch_size)
    for batch in tqdm(iterator, total=len(pairs_in) // args.batch_size):

        ## prepare inputs
        sentences_1, sentences_2 = [], []
        for (qid, docid, rank, score) in batch:
            sentences_1.append(queries[qid])
            sentences_2.append(corpus[docid])

        ## prepare features
        features = tokenizer(
                sentences_1, sentences_2,
                padding=True, 
                truncation=True, 
                return_tensors=True
        ).to(model.device)

        ## predict
        with torch.no_grad():
            scores = model(**features).logits

        ## apply new scores
        pairs = list(
                (batch[i][0], batch[i][1], 0, score) for score in scores
        )
        pairs_new += pairs
        print(pairs_new)

    # rerank and merge
    pairs_in_reranked = sorted(pairs_new, lambda x: x[-1]


