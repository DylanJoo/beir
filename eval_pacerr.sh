# for name in arguana climate-fever dbpedia-entity fiqa nfcorpus scidocs scifact trec-covid webis-touche2020;do

data_dir=/work/jhju/beir-runs
data_dir=/home/jhju/pyserini/topics-and-qrels

pseudo_q=$1
objective=$2
for name in scidocs;do
    for run in run.pacerr.top100/*$pseudo_q*$objective*;do
        echo ${run##*/}
        ~/trec_eval-9.0.7/trec_eval \
            -c -m ndcg_cut.10 \
            $data_dir/qrels.beir-v1.0.0-$name.test.txt $run \
            | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
    done
done

