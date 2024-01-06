# for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do
data_dir=readqg-flan-t5-readqg-calibrate
name=scidocs
python reranking/pacerr_cross_encoder_train.py \
    --dataset datasets/$name \
    --pseudo_queries readqg-flan-t5-readqg-calibrate/$name/calibrate_margin_ibn_dd-20000.jsonl \
    --output_path pace_rr_checkpoints/$name/testing \
    --model_name cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --batch_size 32 \
    --num_epochs 4 \
    --device cuda
