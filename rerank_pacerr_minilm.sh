mkdir -p run.pacerr.top100

setting=$1
variant=$2
# for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do

for name in scidocs;do
    for model in checkpoints/pacerr_minilm$variant/$name/$setting*;do
        echo "Data: " $variant " | Objectives: " ${model##*/}
        python reranking/cross_encoder_predict.py \
            --dataset datasets/$name \
            --input_run /work/jhju/beir-runs/run.beir.bm25-multifield.$name.txt \
            --output_run run.pacerr.top100/run.beir.${model##*/}${variant}.$name.txt \
            --top_k 100 \
            --model_name $model \
            --batch_size 100 \
            --device cuda
    done
done
