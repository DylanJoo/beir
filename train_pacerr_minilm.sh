# # pointwise ranking # bad, try combined
# for name in scidocs; do
#     # data_dir=readqg-flan-t5-readqg-calibrate
#     data_dir=readqg-flan-t5-readqg-baseline
#
#     for file in $data_dir/$name/*jsonl;do
#         setting=${file/.jsonl/}
#         setting=${setting##*/}
#         setting=${setting%.*}
#
#         python reranking/cross_encoder_train.py \
#             --dataset datasets/$name \
#             --pseudo_queries $file \
#             --output_path checkpoints/pacerr_minilm_combined_v1/$name/$setting \
#             --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#             --batch_size 32 \
#             --num_epochs 1 \
#             --learning_rate 1e-5 \
#             --filtering '{"name": "boundary", "num": 1}' \
#             --document_centric \
#             --objective combined-v1 \
#             --device cuda
#     done
# done

# Pairwise
variant=_hinge
for name in scidocs;do
    data_dir=readqg-flan-t5-readqg-calibrate
    # data_dir=readqg-flan-t5-readqg-baseline

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
            --device cuda
    done
done

variant=_lce
for name in scidocs;do
    data_dir=readqg-flan-t5-readqg-calibrate
    # data_dir=readqg-flan-t5-readqg-baseline

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
            --device cuda
    done
done
