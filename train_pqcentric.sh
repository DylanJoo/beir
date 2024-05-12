## THIS NEEDS TO BE REVISED BEFORE USE. use run_train.sh instead
CUDA_VISIBLE_DEVICES=1
variant=_groupwise_bce_hard-hinge_QQ

# [NOTE] this is not exactly the pacerr, still independent
for data in baseline calibrate;do
    # original data_dir
    data_dir=/work/jhju/readqg-flan-t5-readqg-$data
    # data_dir=/work/jhju/readqg-results/

    for name in scidocs;do
        for file in $data_dir/${name}/${data}*jsonl;do
            setting=${file/.jsonl/}
            setting=${setting##*/}
            setting=${setting%.*}
            qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 
            run_bm25=/work/jhju/beir-runs/run.beir.bm25-multifield.$name.txt

            python reranking/cross_encoder_train.py \
                --dataset datasets/$name \
                --pseudo_queries $file \
                --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
                --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
                --batch_size 8 \
                --max_length 384 \
                --num_epochs 2 \
                --learning_rate 7e-6 \
                --do_eval \
                --qrels $qrels \
                --run_bm25 $run_bm25 \
                --filtering '{"name": "top_bottom", "n1": 1, "n2": 1}' \
                --query_centric \
                --objective_qc groupwise_bce_hard \
                --document_centric \
                --objective_dc groupwise_hinge \
                --change_dc_to_qq \
                --margin 1 \
                --device cuda
        done
    done
done
