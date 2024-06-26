variant=_qc_groupwise_bce_hard
for data in baseline calibrate;do
    data_dir=/work/jhju/readqg-flan-t5-readqg-$data
    for name in scidocs;do
        for file in $data_dir/$name/*jsonl;do
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
                --filtering '{"name": "top", "num": 1}' \
                --query_centric \
                --objective_qc groupwise_bce_hard \
                --device cuda
        done
    done
done
