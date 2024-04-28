mkdir -p run.pacerr.top100

data_dir=/work/jhju/beir-runs
pseudo_q=$1
objective=$2
for name in scidocs;do
    for model in checkpoints/pacerr_minilm$objective/$name/$pseudo_q*;do
        for epoch in 0 1;do
            echo "Pseudo data: " ${model##*/} ${epoch}" | Objective: " $objective
            python reranking/cross_encoder_predict.py \
                --dataset datasets/$name \
                --input_run $data_dir/run.beir.bm25-multifield.$name.txt \
                --output_run run.pacerr.top100/run.beir.${model##*/}${objective}.ep${epoch}.$name.txt \
                --top_k 100 \
                --model_name $model/$epoch \
                --batch_size 100 \
                --device cuda
            echo -e
        done
    done
done

# for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do
