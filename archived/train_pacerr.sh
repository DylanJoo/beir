# for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do
for name in scidocs; do
    data_dir=readqg-flan-t5-readqg-calibrate

    for file in $data_dir/$name/*jsonl;do
        setting=${file/.jsonl/}
        setting=${setting##*/}
        setting=${setting%.*}

        python reranking/pacerr_cross_encoder_train.py \
            --dataset datasets/$name \
            --pseudo_queries $file/$name/$setting.jsonl \
            --output_path checkpoints/pacerr_minilm/$name/$setting \
            --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --batch_size 32 \
            --num_epochs 4 \
            --device cuda
    done
done
