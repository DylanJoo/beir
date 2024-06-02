# runing exps

# 32314 cuda 1
CUDA_VISIBLE_DEVICES=1

model_dir=/work/jhju/oodrerank.readqg.others
temp_dir=/work/jhju/temp

# for name in scidocs;do
for name in arguana fiqa nfcorpus scifact scidocs;do

    python reranking/cross_encoder_train_inpars.py \
        --pseudo_queries ${temp_dir}/${name}.jsonl \
        --output_path ${model_dir}/pacerr_minilm_inpars/$name/inpars_v2 \
        --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --batch_size 8 \
        --max_length 384 \
        --num_epochs 2 \
        --learning_rate 7e-6 \
        --filtering '{"name": "top", "num": 1}' \
        --query_centric \
        --objective_qc groupwise_bce_hard \
        --margin 0 \
        --device cuda \
        --save_last 
done
