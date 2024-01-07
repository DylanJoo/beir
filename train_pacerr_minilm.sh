for name in scidocs; do
    data_dir=readqg-flan-t5-readqg-calibrate

    for file in $data_dir/$name/*jsonl;do
        setting=${file/.jsonl/}
        setting=${setting##*/}
        setting=${setting%.*}

        python reranking/pacerr_cross_encoder_train.py \
            --dataset datasets/$name \
            --pseudo_queries $file \
            --output_path checkpoints/pacerr_minilm/$name/$setting \
            --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --batch_size 32 \
            --num_epochs 1 \
            --filtering '{"name": "boundary", "num": 1}' \
            --device cuda
    done
done
