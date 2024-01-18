# # pointwise bce
# variant=_pointwise_bce
# for data in baseline calibrate unlikelihood;do
#     data_dir=readqg-flan-t5-readqg-$data 
#
#     for name in scidocs; do
#
#         for file in $data_dir/$name/*jsonl; do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=qrels/qrels.beir-v1.0.0-$name.test.txt
#
#             echo $setting
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/beir/$name \
#                 --pseudo_queries $file \
#                 --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#                 --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#                 --batch_size 32 \
#                 --num_epochs 1 \
#                 --learning_rate 7e-6 \
#                 --do_eval \
#                 --qrels $qrels \
#                 --filtering '{"name": "boundary", "num": 1}' \
#                 --document_centric \
#                 --objective $variant \
#                 --device cuda
#         done
#     done
# done


# pointwise mse
# variant=_pointwise_mse
# for data in baseline calibrate unlikelihood;do
#     data_dir=readqg-flan-t5-readqg-$data 
#
#     for name in scidocs; do
#
#         for file in $data_dir/$name/*jsonl; do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=qrels/qrels.beir-v1.0.0-$name.test.txt
#
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/beir/$name \
#                 --pseudo_queries $file \
#                 --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#                 --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#                 --batch_size 32 \
#                 --num_epochs 1 \
#                 --learning_rate 7e-6 \
#                 --do_eval \
#                 --qrels $qrels \
#                 --filtering '{"name": "boundary", "num": 1}' \
#                 --document_centric \
#                 --objective $variant \
#                 --device cuda
#         done
#     done
# done

# Distillation mse 
# variant=_distillation_mse
# for data in baseline calibrate unlikelihood;do
#     data_dir=readqg-flan-t5-readqg-$data 
#
#     for name in scidocs; do
#
#         for file in $data_dir/$name/*jsonl; do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=qrels/qrels.beir-v1.0.0-$name.test.txt
#
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/beir/$name \
#                 --pseudo_queries $file \
#                 --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#                 --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#                 --batch_size 4 \
#                 --num_epochs 1 \
#                 --learning_rate 7e-6 \
#                 --do_eval \
#                 --qrels $qrels \
#                 --filtering '{"name": "all"}' \
#                 --document_centric \
#                 --objective $variant \
#                 --device cuda
#         done
#     done
# done


# variant=_qc_groupwise_ce_all
# for data in baseline calibrate unlikelihood;do
#     data_dir=readqg-flan-t5-readqg-$data
#
#     for name in scidocs;do
#         for file in $data_dir/$name/*jsonl;do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=qrels/qrels.beir-v1.0.0-$name.test.txt 
#
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/beir/$name \
#                 --pseudo_queries $file \
#                 --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#                 --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#                 --batch_size 8 \
#                 --num_epochs 1 \
#                 --learning_rate 7e-6 \
#                 --do_eval \
#                 --qrels $qrels \
#                 --filtering '{"name": "top", "num": 1}' \
#                 --query_centric \
#                 --objective $variant \
#                 --margin 1 \
#                 --device cuda:2
#         done
#     done
# done

variant=_bi_groupwise_ce_all
for data in baseline calibrate unlikelihood;do
    data_dir=readqg-flan-t5-readqg-$data

    for name in scidocs;do
        for file in $data_dir/$name/*jsonl;do
            setting=${file/.jsonl/}
            setting=${setting##*/}
            setting=${setting%.*}
            qrels=qrels/qrels.beir-v1.0.0-$name.test.txt 

            python reranking/cross_encoder_train.py \
                --dataset datasets/beir/$name \
                --pseudo_queries $file \
                --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
                --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
                --batch_size 4 \
                --num_epochs 1 \
                --learning_rate 7e-6 \
                --do_eval \
                --qrels $qrels \
                --filtering '{"name": "boundary", "num": 1}' \
                --query_centric \
                --document_centric \
                --objective $variant \
                --device cuda:1
        done
    done
done
