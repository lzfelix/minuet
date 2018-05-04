#!/bin/bash

wget -O glove.zip http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.zip glove.6B.300d.txt
rm glove.zip

