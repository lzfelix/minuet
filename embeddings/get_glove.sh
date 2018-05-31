#!/bin/bash

wget -O glove.zip http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.zip glove.6B.300d.txt
rm glove.zip

echo "Converting to word2vec format"
python -m gensim.scripts.glove2word2vec -i glove.6B.300d.txt -o wglove.6B.300d.txt

echo "Converting to gensim format"
python -c "from gensim.models import KeyedVectors as kv; kv.load_word2vec_format('wglove.6B.300d.txt').save('wglove.6B.300d.bin')"

rm wglove.6B.300d.txt
rm glove.6B.300d.txt

