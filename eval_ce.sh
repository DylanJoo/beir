# | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

echo trec-covid
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  /home/jhju/pyserini/topics-and-qrels/qrels.beir-v1.0.0-trec-covid.test.txt \
  run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.trec-covid.txt 

echo NFCorpus
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  /home/jhju/pyserini/topics-and-qrels/qrels.beir-v1.0.0-nfcorpus.test.txt \
  run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.nfcorpus.txt

echo FiQA
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  /home/jhju/pyserini/topics-and-qrels/qrels.beir-v1.0.0-fiqa.test.txt \
  run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.fiqa.txt

echo arguana
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  /home/jhju/pyserini/topics-and-qrels/qrels.beir-v1.0.0-arguana.test.txt \
  run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.arguana.txt

echo Touche
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  /home/jhju/pyserini/topics-and-qrels/qrels.beir-v1.0.0-webis-touche2020.test.txt \
  run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.webis-touche2020.txt

echo DBPedia
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  /home/jhju/pyserini/topics-and-qrels/qrels.beir-v1.0.0-dbpedia-entity.test.txt \
  run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.dbpedia-entity.txt

echo Scidocs
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  /home/jhju/pyserini/topics-and-qrels/qrels.beir-v1.0.0-scidocs.test.txt \
  run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.scidocs.txt

echo climate-fever
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  /home/jhju/pyserini/topics-and-qrels/qrels.beir-v1.0.0-climate-fever.test.txt \
  run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.climate-fever.txt

echo scifact
~/trec_eval-9.0.7/trec_eval \
  -c -m ndcg_cut.10 \
  /home/jhju/pyserini/topics-and-qrels/qrels.beir-v1.0.0-scifact.test.txt \
  run.ce.top100/run.beir.ms-marco-MiniLM-L-6-v2.scifact.txt
