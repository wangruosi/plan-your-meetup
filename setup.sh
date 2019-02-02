if [ ! -d app ]; then
   mkdir app
fi

if [ ! -d doc ]; then
   mkdir doc
fi

if [ ! -d results ]; then
   mkdir results
fi

if [ ! -d data ]; then
   mkdir data
fi

cd data
 

if [ ! -d glove.6B ]; then
   mkdir glove.6B
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   mv glove.6B.*.txt glove.6B
   rm glove.6B.zip
fi
cd ..

