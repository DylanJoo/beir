QCDC=groupwise_bce
variant=${QCDC}
# for data in baseline calibrate;do
for data in calibrate;do
    data_dir=/work/jhju/readqg-flan-t5-readqg-$data

    for name in scidocs;do
        for file in $data_dir/$name/*jsonl;do
            setting=${file/.jsonl/}
            setting=${setting##*/}
            setting=${setting%.*}
            qrels=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 
            bm25=/work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt 

            python reranking/cross_encoder_train_test.py \
                --dataset datasets/$name \
                --pseudo_queries $file \
                --output_path checkpoints/pacerr_minilm$variant/$name/$setting \
                --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
                --batch_size 16 \
                --max_length 384 \
                --num_epochs 2 \
                --learning_rate 7e-6 \
                --do_eval \
                --qrels $qrels \
                --filtering '{"name": "top_bottom", "n1": 1, "n2": 1}' \
                --objective groupwise_bce \
                --save_last \
                --device cuda 
        done
    done
done

