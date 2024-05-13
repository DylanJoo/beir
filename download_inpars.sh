temp_dir=/work/jhju/temp
mkdir -p $temp_dir

# download generated queries
wget https://huggingface.co/datasets/inpars/generated-data/resolve/main/arguana/triples_train_v2.tsv?download=true -O /${temp_dir}/arguana.inpars
wget https://huggingface.co/datasets/inpars/generated-data/resolve/main/fiqa/triples_train_v2.tsv?download=true -O /${temp_dir}/fiqa.inpars
wget https://huggingface.co/datasets/inpars/generated-data/resolve/main/nfcorpus/triples_train_v2.tsv?download=true -O /${temp_dir}/nfcorpus.inpars
wget https://huggingface.co/datasets/inpars/generated-data/resolve/main/scifact/triples_train_v2.tsv?download=true -O /${temp_dir}/scifact.inpars
wget https://huggingface.co/datasets/inpars/generated-data/resolve/main/scidocs/triples_train_v2.tsv?download=true -O /${temp_dir}/scidocs.inpars

# shuffle and truncate with the corpus size
cat /${temp_dir}/arguana.inpars | shuf | head -n 8674 > /work/jhju/arguana.shuf.inpars
cat /${temp_dir}/fiqa.inpars | shuf | head -n 57638 > /${temp_dir}/fiqa.shuf.inpars
cat /${temp_dir}/nfcorpus.inpars | shuf |  head -n 3633  > /${temp_dir}/nfcorpus.shuf.inpars
cat /${temp_dir}/scidocs.inpars | shuf | head -n 25657 > /${temp_dir}/scidocs.shuf.inpars
cat /${temp_dir}/scifact.inpars | shuf | head -n 5183 > /${temp_dir}/scifact.shuf.inpars

wc -l /${temp_dir}/*shuf*
