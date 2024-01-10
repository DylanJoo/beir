data_dir=/work/jhju/readqg-flan-t5-readqg-unlikelihood # 32718
data_dir=/work/jhju/readqg-flan-t5-readqg-baseline # 31949
data_dir=/work/jhju/readqg-flan-t5-readqg-calibrate # 32718

for name in scidocs; do

    for file in $data_dir/$name/*jsonl;do
        setting=${file/.jsonl/}
        setting=${setting##*/}
        setting=${setting%.*}

        python reranking/cross_encoder_train.py \
            --dataset datasets/$name \
            --pseudo_queries $file \
            --output_path checkpoints/pacerr_minilm_pointwise_bce/$name/$setting \
            --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --batch_size 32 \
            --num_epochs 1 \
            --learning_rate 1e-5 \
            --filtering '{"name": "boundary", "num": 1}' \
            --document_centric \
            --objective pointwise-bce \
            --device cuda:0
    done
done

for name in scidocs; do

    for file in $data_dir/$name/*jsonl;do
        setting=${file/.jsonl/}
        setting=${setting##*/}
        setting=${setting%.*}

        python reranking/cross_encoder_train.py \
            --dataset datasets/$name \
            --pseudo_queries $file \
            --output_path checkpoints/pacerr_minilm_pointwise_mse/$name/$setting \
            --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --batch_size 32 \
            --num_epochs 1 \
            --learning_rate 1e-5 \
            --filtering '{"name": "boundary", "num": 1}' \
            --document_centric \
            --objective pointwise-mse \
            --device cuda:0
    done
done

variant=_pairwise_hinge
for name in scidocs; do

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
            --learning_rate 1e-5 \
            --filtering '{"name": "boundary", "num": 1}' \
            --document_centric \
            --objective pairwise-hinge \
            --device cuda:0
    done
done

variant=_pairwise_lce
for name in scidocs; do

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
            --learning_rate 1e-5 \
            --filtering '{"name": "boundary", "num": 1}' \
            --document_centric \
            --objective pairwise-lce \
            --device cuda:0
    done
done

