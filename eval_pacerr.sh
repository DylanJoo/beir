# for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do
for name in scidocs;do
    for result in run.pacerr.top100/run.beir.*.$name.txt; do
        echo $result
        ~/trec_eval-9.0.7/trec_eval \
            -c -m ndcg_cut.10 \
            /work/jhju/beir-runs/qrels.beir-v1.0.0-$name.test.txt \
            $result | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
    done
done

