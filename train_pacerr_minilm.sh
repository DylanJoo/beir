data_dir=readqg-flan-t5-readqg-calibrate

# pointwise
for name in scidocs; do

    for file in $data_dir/$name/*jsonl;do
        setting=${file/.jsonl/}
        setting=${setting##*/}
        setting=${setting%.*}

        python reranking/cross_encoder_train.py \
            --dataset datasets/$name \
            --pseudo_queries $file \
            --output_path checkpoints/pacerr_minilm/$name/$setting \
            --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --batch_size 32 \
            --num_epochs 1 \
            --learning_rate 7e-6 \
            --filtering '{"name": "boundary", "num": 1}' \
            --document_centric \
            --objective pointwise-bce \
            --device cuda
    done
done

# # pairwise
# variant=_hinge
# for name in scidocs;do
#
#     for file in $data_dir/$name/*jsonl;do
#         setting=${file/.jsonl/}
#         setting=${setting##*/}
#         setting=${setting%.*}
#
#         python reranking/cross_encoder_train.py \
#             --dataset datasets/$name \
#             --pseudo_queries $file \
#             --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#             --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#             --batch_size 32 \
#             --num_epochs 1 \
#             --learning_rate 7e-6 \
#             --filtering '{"name": "boundary", "num": 1}' \
#             --document_centric \
#             --objective pairwise-hinge \
#             --device cuda
#     done
# done
#
# variant=_lce
# for name in scidocs;do
#
#     for file in $data_dir/$name/*jsonl;do
#         setting=${file/.jsonl/}
#         setting=${setting##*/}
#         setting=${setting%.*}
#
#         python reranking/cross_encoder_train.py \
#             --dataset datasets/$name \
#             --pseudo_queries $file \
#             --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#             --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#             --batch_size 32 \
#             --num_epochs 1 \
#             --learning_rate 7e-6 \
#             --filtering '{"name": "boundary", "num": 1}' \
#             --document_centric \
#             --objective pairwise-lce \
#             --device cuda
#     done
# done

# combined pointwise and pairwise
variant=_combined_v1
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
            --objective combined-v1 \
            --device cuda
    done
done

variant=_combined_v2
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
            --objective combined-v2 \
            --device cuda
    done
done
