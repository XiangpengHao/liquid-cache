mkdir -p raw_gz/

# Download all files
wget https://datasets.imdbws.com/name.basics.tsv.gz -O raw_gz/name.basics.tsv.gz      
wget https://datasets.imdbws.com/title.akas.tsv.gz -O raw_gz/title.akas.tsv.gz
wget https://datasets.imdbws.com/title.basics.tsv.gz -O raw_gz/title.basics.tsv.gz
wget https://datasets.imdbws.com/title.crew.tsv.gz -O raw_gz/title.crew.tsv.gz
wget https://datasets.imdbws.com/title.episode.tsv.gz -O raw_gz/title.episode.tsv.gz
wget https://datasets.imdbws.com/title.principals.tsv.gz -O raw_gz/title.principals.tsv.gz
wget https://datasets.imdbws.com/title.ratings.tsv.gz -O raw_gz/title.ratings.tsv.gz