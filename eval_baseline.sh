data_dir=/work/jhju/beir-runs

# for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do
for model in ms-marco-MiniLM-L-6-v2;do
# for model in ms-marco-MiniLM-L-6-v2 monot5-base-msmarco-10k;do
    echo $model 
    # for name in arguana climate-fever scidocs trec-covid webis-touche2020;do
    for name in scidocs;do
        echo -e $name " ";
        ~/trec_eval-9.0.7/trec_eval \
            -c -m ndcg_cut.10 \
            $data_dir/qrels.beir-v1.0.0-$name.test.txt $run \
            run.ce.top100/run.beir.$model.$name.txt \
            | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
    done
    echo -e
done

