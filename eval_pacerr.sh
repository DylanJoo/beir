# for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do
# for name in scidocs;do
#     for model in checkpoints/pacerr_minilm/$name/*;do
for name in scidocs;do
    for crossencoder in checkpoints/pacerr_minilm/$name/*;do
        echo ${crossencoder##*/}
        ~/trec_eval-9.0.7/trec_eval \
            -c -m ndcg_cut.10 \
            /home/jhju/pyserini/topics-and-qrels/qrels.beir-v1.0.0-$name.test.txt \
            run.pacerr.top100/run.beir.${crossencoder##*/}.$name.txt \
            | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
    done
    echo -e
done
