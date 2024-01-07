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

# [bi-directional inputs]
# calibrate_margin_ibn_dd-20000 # 0.0636
# calibrate_margin_ibn_na-20000 # 0.0978
# calibrate_margin_ibn_qd-20000 # 0.0474
# calibrate_rank_ibn_dd-20000   # 0.0860
# calibrate_rank_ibn_na-20000   # 0.0820
# calibrate_rank_ibn_qd-20000   # 0.0485

# [uni-directional inputs]
# calibrate_margin_ibn_dd-20000
# calibrate_margin_ibn_na-20000
# calibrate_margin_ibn_qd-20000
# calibrate_rank_ibn_dd-20000
# calibrate_rank_ibn_na-20000
# calibrate_rank_ibn_qd-20000
