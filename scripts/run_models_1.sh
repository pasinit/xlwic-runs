cd ..
PYTHONPATH=src python src/train.py \
    data.target_identification=char_2_token\
    model.subword_combiner=mean
    model.words_combiner=cat

PYTHONPATH=src python src/train.py \
    data.target_identification=char_2_token\
    model.subword_combiner=mean
    model.words_combiner=sum

PYTHONPATH=src python src/train.py \
    data.target_identification=char_2_token\
    model.subword_combiner=sum
    model.words_combiner=cat

PYTHONPATH=src python src/train.py \
    data.target_identification=char_2_token\
    model.subword_combiner=mean
    model.words_combiner=sum