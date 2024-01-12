data_dir=readqg-flan-t5-readqg-baseline

objective=_pairwise_hinge
for name in scidocs;do

    for file in $data_dir/$name/*jsonl;do
        setting=${file/.jsonl/}
        setting=${setting##*/}
        setting=${setting%.*}

        python reranking/cross_encoder_train.py \
            --dataset datasets/$name \
            --pseudo_queries $file \
            --output_path checkpoints/pacerr_minilm$objective/$name/$setting \
            --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --batch_size 32 \
            --num_epochs 1 \
            --learning_rate 7e-6 \
            --filtering '{"name": "boundary", "num": 1}' \
            --document_centric \
            --objective $objective \
            --device cuda
    done
done

# objective=_lce
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
#             --output_path checkpoints/pacerr_minilm$objective/$name/$setting \
#             --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#             --batch_size 32 \
#             --num_epochs 1 \
#             --learning_rate 7e-6 \
#             --filtering '{"name": "boundary", "num": 1}' \
#             --document_centric \
#             --objective $objective \
#             --device cuda
#     done
# done

# groupwise
# objective=_groupwise_hinge
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
#             --output_path checkpoints/pacerr_minilm$objective/$name/$setting \
#             --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#             --batch_size 32 \
#             --num_epochs 1 \
#             --learning_rate 7e-6 \
#             --filtering '{"name": "boundary", "num": 3}' \
#             --document_centric \
#             --objective $objective \
#             --device cuda
#     done
# done

# groupwise
# objective=_groupwise_lce
# objective=_groupwise_hinge
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
#             --output_path checkpoints/pacerr_minilm$objective/$name/$setting \
#             --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#             --batch_size 32 \
#             --num_epochs 1 \
#             --learning_rate 7e-6 \
#             --filtering '{"name": "none"}' \
#             --document_centric \
#             --objective $objective \
#             --device cuda
#     done
# done
