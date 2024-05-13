export CUDA_VISIBLE_DEVICES=0
mkdir -p run.ce.top100

for name in nfcorpus fiqa arguana scidocs scifact;do
    python reranking/monot5_reranker_predict.py \
        --dataset datasets/$name \
        --input_run run.bm25/run.beir.bm25-multifield.$name.txt \
        --output_run run.ce.top100/run.beir.monot5-base-msmarco-10k.$name.txt \
        --top_k 100 \
        --model_name castorini/monot5-base-msmarco-10k \
        --batch_size 100 \
        --device cuda
done
