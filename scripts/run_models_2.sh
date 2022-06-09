cd ..
PYTHONPATH=src python src/train.py \
    data.target_identification=left_right\
    model.subword_combiner=mean
    model.words_combiner=cat

PYTHONPATH=src python src/train.py \
    data.target_identification=left_right\
    model.subword_combiner=mean
    model.words_combiner=sum

PYTHONPATH=src python src/train.py \
    data.target_identification=left_right\
    model.subword_combiner=sum
    model.words_combiner=cat

PYTHONPATH=src python src/train.py \
    data.target_identification=left_right\
    model.subword_combiner=mean
    model.words_combiner=sum