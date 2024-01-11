data_dir=readqg-flan-t5-readqg-calibrate

variant=_pairwise_hinge
for name in scidocs;do

    for file in $data_dir/$name/*jsonl;do
        setting=${file/.jsonl/}
        setting=${setting##*/}
        setting=${setting%.*}

        python reranking/cross_encoder_train.py \
            --dataset datasets/$name \
            --pseudo_queries $file \
            --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
            --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --batch_size 32 \
            --num_epochs 1 \
            --learning_rate 7e-6 \
            --filtering '{"name": "boundary", "num": 1}' \
            --document_centric \
            --objective $variant \
            --device cuda
    done
done

variant=_pairwise_lce
for name in scidocs;do

    for file in $data_dir/$name/*jsonl;do
        setting=${file/.jsonl/}
        setting=${setting##*/}
        setting=${setting%.*}

        python reranking/cross_encoder_train.py \
            --dataset datasets/$name \
            --pseudo_queries $file \
            --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
            --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --batch_size 32 \
            --num_epochs 1 \
            --learning_rate 7e-6 \
            --filtering '{"name": "boundary", "num": 1}' \
            --document_centric \
            --objective $variant \
            --device cuda
    done
done
