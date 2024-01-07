export CUDA_VISIBLE_DEVICES=1
for name in scidocs; do
    data_dir=readqg-flan-t5-readqg-calibrate

    for file in $data_dir/$name/*jsonl;do
        setting=${file/.jsonl/}
        setting=${setting##*/}
        setting=${setting%.*}

        python reranking/pacerr_cross_encoder_train.py \
            --dataset datasets/$name \
            --pseudo_queries $file \
            --output_path checkpoints/pacerr_monot5/$name/$setting \
            --model_name castorini/monot5-base-msmarco-10k \
            --batch_size 8 \
            --num_epochs 4 \
            --filtering '{"name": "boundary", "num": 1}' \
            --device cuda
    done
done
