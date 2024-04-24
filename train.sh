# 1. QQ classification
# QC: Query-InBatchDocNegative; DC: Query-GeneratedHardQueryNegative
QC=groupwise_bce_hard
DC=groupwise_bce
variant=${QC}-${DC}
# for data in baseline calibrate;do
for data in calibrate;do
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
                --output_path checkpoints/pacerr_minilm_$variant/$name/$setting \
                --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
                --batch_size 8 \
                --num_epochs 2 \
                --learning_rate 7e-6 \
                --do_eval \
                --qrels $qrels \
                --run_bm25 $run_bm25 \
                --filtering '{"name": "top_bottom", "n1": 1, "n2": 1}' \
                --query_centric \
                --objective_qc groupwise_bce_hard \
                --document_centric \
                --objective_dc groupwise_bce \
                --change_dc_to_qq \
                --save_last \
                --device cuda
        done
    done
done

# 2. Enhance document repl from negative query -- BCE version
# QC: Query-InBatchDocNegative; DC: GeneratedHardQueryNegative-Doc (as negative)

# 3. Enhance document repl from negative query -- BCE version
# QC: Query-InBatchDocNegative; DC: GeneratedHardQueryNegative-Doc (as positive)
