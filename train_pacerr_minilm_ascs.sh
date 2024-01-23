# variant=_pairwise_hinge
# for data in baseline calibrate unlikelihood;do
#     data_dir=/work/jhju/readqg-flan-t5-readqg-$data
#     for name in scidocs;do
#         for file in $data_dir/$name/*jsonl;do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 
#
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/$name \
#                 --pseudo_queries $file \
#                 --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#                 --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#                 --batch_size 16 \
#                 --num_epochs 1 \
#                 --learning_rate 7e-6 \
#                 --do_eval \
#                 --qrels $qrels \
#                 --filtering '{"name": "boundary", "num": 1}' \
#                 --objective $variant \
#                 --margin 1 \
#                 --device cuda
#         done
#     done
# done

# variant=_groupwise_hinge_v1
# for data in baseline calibrate unlikelihood;do
#     data_dir=/work/jhju/readqg-flan-t5-readqg-$data
#     for name in scidocs;do
#         for file in $data_dir/$name/*jsonl;do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 
#
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/$name \
#                 --pseudo_queries $file \
#                 --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#                 --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#                 --batch_size 4 \
#                 --num_epochs 1 \
#                 --learning_rate 7e-6 \
#                 --do_eval \
#                 --qrels $qrels \
#                 --filtering '{"name": "all"}' \
#                 --objective $variant \
#                 --margin 1 \
#                 --device cuda
#         done
#     done
# done

# variant=_pairwise_ce
# for data in baseline calibrate unlikelihood;do
#     data_dir=/work/jhju/readqg-flan-t5-readqg-$data
#     for name in scidocs;do
#         for file in $data_dir/$name/*jsonl;do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 
#
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/$name \
#                 --pseudo_queries $file \
#                 --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#                 --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#                 --batch_size 16 \
#                 --num_epochs 1 \
#                 --learning_rate 7e-6 \
#                 --do_eval \
#                 --qrels $qrels \
#                 --filtering '{"name": "boundary", "num": 1}' \
#                 --objective $variant \
#                 --margin 1 \
#                 --document_centric \
#                 --device cuda
#         done
#     done
# done

# variant=_groupwise_ce
# for data in baseline calibrate unlikelihood;do
#     data_dir=/work/jhju/readqg-flan-t5-readqg-$data
#     for name in scidocs;do
#         for file in $data_dir/$name/*jsonl;do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 
#
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/$name \
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

# variant=_groupwise_ce_pair
# for data in baseline calibrate unlikelihood;do
#     data_dir=/work/jhju/readqg-flan-t5-readqg-$data
#     for name in scidocs;do
#         for file in $data_dir/$name/*jsonl;do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 
#
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/$name \
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

# variant=_groupwise_ce_v1
# for data in baseline calibrate unlikelihood;do
#     data_dir=/work/jhju/readqg-flan-t5-readqg-$data
#     for name in scidocs;do
#         for file in $data_dir/$name/*jsonl;do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 
#
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/$name \
#                 --pseudo_queries $file \
#                 --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#                 --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#                 --batch_size 4 \
#                 --num_epochs 1 \
#                 --learning_rate 7e-6 \
#                 --do_eval \
#                 --qrels $qrels \
#                 --filtering '{"name": "all"}' \
#                 --objective $variant \
#                 --margin 1 \
#                 --document_centric \
#                 --device cuda
#         done
#     done
# done

# variant=_qc_groupwise_ce_all_boundary
# for data in baseline calibrate unlikelihood;do
#     data_dir=/work/jhju/readqg-flan-t5-readqg-$data
#
#     for name in scidocs;do
#         for file in $data_dir/$name/*jsonl;do
#             setting=${file/.jsonl/}
#             setting=${setting##*/}
#             setting=${setting%.*}
#             qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 
#
#             python reranking/cross_encoder_train.py \
#                 --dataset datasets/$name \
#                 --pseudo_queries $file \
#                 --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
#                 --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
#                 --batch_size 6 \
#                 --num_epochs 1 \
#                 --learning_rate 7e-6 \
#                 --do_eval \
#                 --qrels $qrels \
#                 --filtering '{"name": "boundary", "num": 1}' \
#                 --query_centric \
#                 --objective $variant \
#                 --device cuda
#         done
#     done
# done

variant=_bi_groupwise_ce-groupwise_hinge_v1
for data in baseline calibrate unlikelihood;do
    data_dir=/work/jhju/readqg-flan-t5-readqg-$data

    for name in scidocs;do
        for file in $data_dir/$name/*jsonl;do
            setting=${file/.jsonl/}
            setting=${setting##*/}
            setting=${setting%.*}
            qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 

            python reranking/cross_encoder_train.py \
                --dataset datasets/$name \
                --pseudo_queries $file \
                --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
                --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
                --batch_size 4 \
                --num_epochs 1 \
                --learning_rate 7e-6 \
                --do_eval \
                --qrels $qrels \
                --filtering '{"name": "all"}' \
                --document_centric \
                --objective_dc groupwise_hinge_v1 \
                --query_centric \
                --objective_qc groupwise_ce \
                --device cuda
        done
    done
done
