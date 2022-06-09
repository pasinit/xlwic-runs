from argparse import ArgumentParser
import yaml
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--words_combiner', choices=['cat', 'sum'], required=True)
    parser.add_argument('--subword_combiner', choices=['mean', 'sum'], required = True)
    parser.add_argument('--target_identification', choices=['left_right', 'char_2_token'], required = True)
    args = parser.parse_args()
    with open('conf/model/default_model.yaml') as reader:
        model_conf = yaml.safe_load(reader)
    with open('conf/data/default_data.yaml') as reader:
        data_conf = yaml.safe_load(reader)
    
    model_conf['words_combiner'] = args.words_combiner
    model_conf['subword_combiner'] = args.subword_combiner
    data_conf['target_identification'], args.target_identification
    with open('conf/model/default_model.yaml', 'w') as writer:
        yaml.dump(model_conf, writer)

    with open('conf/data/default_data.yaml', 'w') as writer:
        yaml.dump(data_conf, writer)
    os.system('PYTHONPATH=src:. python src/train.py')

    