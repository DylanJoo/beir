# runing exps

# 32314 cuda 0
# CUDA_VISIBLE_DEVICES=0
# variant=_inpars


model_dir=/work/jhju/oodrerank.readqg.inpars
data_dir=/work/jhju/inpars-results/

for data in calibrate baseline;do

    # for name in scidocs;do
    for name in arguana fiqa nfcorpus scifact scidocs;do

        python reranking/cross_encoder_train.py \
            --dataset datasets/$name \
            --pseudo_queries /content/${name}.jsonl \
            --output_path ${model_dir}/pacerr_minilm$variant/$name/inpars \
            --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
            --batch_size 8 \
            --max_length 384 \
            --num_epochs 2 \
            --learning_rate 7e-6 \
            --filtering '{"name": "top", "num": 1}' \
            --query_centric \
            --objective_qc groupwise_bce_hard \
            --margin 0 \
            --device cuda
    done
done
