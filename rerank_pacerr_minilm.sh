mkdir -p run.ce.top100

# for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do
for name in scidocs;do
    for model in checkpoints/pacerr_minilm_hinge/$name/$1*;do
        python reranking/cross_encoder_predict.py \
            --dataset datasets/$name \
            --input_run run.bm25/run.beir.bm25-multifield.$name.txt \
            --output_run run.pacerr.top100/run.beir.${model##*/}.$name.txt \
            --top_k 100 \
            --model_name $model \
            --batch_size 100 \
            --device cuda
    done
done
