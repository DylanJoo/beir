export CUDA_VISIBLE_DEVICES=2
mkdir -p run.ce.top100

for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do
    python reranking/cross_encoder_preduct.py \
        --dataset datasets/$name \
        --input_run run.bm25/run.beir.bm25-multifield.$name.txt \
        --output_run run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.$name.txt \
        --top_k 100 \
        --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --batch_size 100 \
        --device cuda
done
