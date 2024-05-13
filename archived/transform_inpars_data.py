from datasets import load_dataset
import json

for dataset_name in ['arguana','fiqa','nfcorpus','scidocs','scifact']:
    d = load_dataset('csv',
                     data_files=f'/work/jhju/temp/{dataset_name}.shuf.inpars',
                     delimiter='\t',
                     column_names=['query', 'positive', 'negative'])['train']

    with open(f'/work/jhju/temp/{dataset_name}.jsonl', 'w') as fout:
        for i, example in enumerate(d):
            fout.write(json.dumps({
                'doc_id': str(i),
                'document': example['positive'],
                'relevance_scores': [1.0],
                'generated_query': [example['query']]
            } )+'\n')
