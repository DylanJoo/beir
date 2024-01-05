python pace_rr/predict.py \
    --dataset datasets/trec-covid \
    --input_run run.bm25/run.beir.bm25-multifield.trec-covid.txt \
    --output_run run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.trec-covid.txt \
    --top_k 100 \
    --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --batch_size 8 \
    --device cuda
