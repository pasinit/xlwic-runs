from re import I
import numpy as np
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import pickle as pkl
BG='xlwic_wn/bulgarian_bg'
ZH='xlwic_wn/chinese_zh'
HR='xlwic_wn/croatian_hr'
DA='xlwic_wn/danish_da'  
NL='xlwic_wn/dutch_nl'
ET='xlwic_wn/estonian_et'
FA='xlwic_wn/farsi_fa'
JA='xlwic_wn/japanese_ja'
KO='xlwic_wn/korean_ko'
FR='xlwic_wikt/french_fr'  
DE='xlwic_wikt/german_de'
IT='xlwic_wikt/italian_it'
XLWIC_TESET_PATHS= [BG, ZH, HR, DA, NL, ET, JA, FA, KO, FR, DE, IT]
CACHE_DIR='/home/tommaso/dev/xlwic/data/xlwic_datasets/.cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
        
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class XLWICDataset(Dataset):
    def __init__(self, data_path, tokenizer:PreTrainedTokenizer, language, split, target_identification, 
    subword_combiner,
    answer_path=None):
        super().__init__()
        self.name = f'xlwic-{language}-{split}'
        self.language = language
        self.examples = examples = self.maybe_load_cache()
        self.subword_combiner = subword_combiner
        if examples is None:
            self.examples = examples = []
            with open(data_path) as lines:
                for line in tqdm(lines, desc=f'loading xlwic-{language} data'):
                    line = line.lower()
                    lemma, pos, idx_start_1, idx_end_1, idx_start_2, idx_end_2, s1, s2, *label = line.strip().split('\t')
                    idx_start_1, idx_end_1, idx_start_2, idx_end_2 = int(idx_start_1), int(idx_end_1), int(idx_start_2), int(idx_end_2)
                    if target_identification == 'left_right':
                        input_ids, indices_mask = self.get_input_ids_left_right(tokenizer, s1, s2, idx_start_1, idx_end_1, idx_start_2, idx_end_2)
                    elif target_identification == 'char_2_token':
                        input_ids, indices_mask = self.get_input_ids_char_to_token(tokenizer, s1, s2, idx_start_1, idx_end_1, idx_start_2, idx_end_2)
                    else: 
                        raise RuntimeError(f"Target identification method {target_identification} not recognised, please choose between \{'left_right', 'char_2_token'}.")
                    example = {'input_ids': input_ids, 'indices_mask': indices_mask}
                    if len(label) > 0:
                        example['label'] = int(label[0])
                    examples.append(example)
            self.dump_to_cache()
        if answer_path is not None:
            with open(answer_path) as lines:
                for i, answer in enumerate(lines):
                    examples[i]['label'] = int(answer)


    def get_input_ids_left_right(self, tokenizer, s1, s2, idx_start_1, idx_end_1, idx_start_2, idx_end_2):
        assert idx_start_1 < idx_end_1
        assert idx_start_2 < idx_end_2
        s1 = s1.replace(u'\u200c', '')
        s2 = s2.replace(u'\u200c', '')
        s1_encodings_left = tokenizer(s1[:idx_start_1])['input_ids'][:-1] # take off eos
        s1_encodings_word = tokenizer(s1[idx_start_1:idx_end_1], add_special_tokens=False)['input_ids']
        s1_encodings_right = tokenizer(s1[idx_end_1:])['input_ids'][1:] # take off bos
        s1_encodings = s1_encodings_left + s1_encodings_word + s1_encodings_right

        s2_encodings_left = tokenizer(s2[:idx_start_2], add_special_tokens=False)['input_ids']
        s2_encodings_word = tokenizer(s2[idx_start_2:idx_end_2], add_special_tokens=False)['input_ids']
        s2_encodings_right = tokenizer(s2[idx_end_2:])['input_ids'][1:] # take off bos
        s2_encodings = s2_encodings_left + s2_encodings_word + s2_encodings_right

            
        target_token_idx_1 = list(range(len(s1_encodings_left), len(s1_encodings_left) + len(s1_encodings_word)))
        target_token_idx_2 = list(range(len(s1_encodings) + len(s2_encodings_left), len(s1_encodings) + len(s2_encodings_left) + len(s2_encodings_word)))
        if self.subword_combiner == 'first':
            target_token_idx_1 = target_token_idx_1[0:1]
            target_token_idx_2 = target_token_idx_2[0:1]

        input_ids = s1_encodings + s2_encodings

        ## Gotta strip off outer and inner spaces for the check mainly for Farsi.
        decoded_word_1 = tokenizer.decode(np.array(input_ids)[target_token_idx_1]).strip().replace(' ', '')
        decoded_word_2 = tokenizer.decode(np.array(input_ids)[target_token_idx_2]).strip().replace(' ', '')
        if not (decoded_word_1 == tokenizer.unk_token or decoded_word_1 == s1[idx_start_1:idx_end_1].strip().replace(' ', '')):
            print(f'[WARNING] selected target word from input_ids and target word from the string are not exactly the same.\nFrom ids: {decoded_word_1}; From string {s1[idx_start_1:idx_end_1]}')
        if not (decoded_word_2 == tokenizer.unk_token or decoded_word_2 == s2[idx_start_2:idx_end_2].strip().replace(' ', '')):
            print(f'[WARNING] selected target word from input_ids and target word from the string are not exactly the same.\nFrom ids: {decoded_word_2}; From string {s2[idx_start_2:idx_end_2]}')
        if decoded_word_2 == tokenizer.unk_token or decoded_word_1 == tokenizer.unk_token:
            print(f'WARNING the target tokens ({s1[idx_start_1:idx_end_1]}, {s2[idx_start_2:idx_end_2]}) are tokenized as UNK')
        indices_mask = np.zeros_like(input_ids, dtype=int)
        indices_mask[target_token_idx_1] = 1
        indices_mask[target_token_idx_2] = 2
        return input_ids, indices_mask

                
    def get_input_ids_char_to_token(self, tokenizer, s1, s2, idx_start_1, idx_end_1, idx_start_2, idx_end_2):
        s1_encodings = tokenizer(s1).encodings[0]
        s2_encodings = tokenizer(s2).encodings[0]
        s1_input_ids = s1_encodings.ids
        target_token_idx_1 = self.get_target_token_indices(s1_encodings.offsets, idx_start_1, idx_end_1)
        target_token_idx_2 = self.get_target_token_indices(s2_encodings.offsets, idx_start_2, idx_end_2)
                    
        target_token_idx_2 += len(s1_input_ids) - 1 # to make up for bos_token in s2_encoding_ids
        s2_input_ids = s2_encodings.ids[1:] # take out the bos_token

        input_ids = s1_input_ids + s2_input_ids
        if self.subword_combiner == 'first':
            target_token_idx_1 = target_token_idx_1[0:1]
            target_token_idx_2 = target_token_idx_2[0:1]
        indices_mask = np.zeros_like(input_ids, dtype=int)
        indices_mask[target_token_idx_1] = 1
        indices_mask[target_token_idx_2] = 2
        return input_ids, indices_mask

    def get_target_token_indices(self, offsets, start_char, end_char):
        for i, pair in enumerate(offsets):
            if start_char >= pair[0] and start_char < pair[1] and sum(pair) > 0:
                j = i+1
                while j < len(offsets) and offsets[j][1] <= end_char and offsets[j][0] == pair[1]:
                    j+=1 
                    continue
                return np.array(list(range(i, j)))
        raise RuntimeError("Cannot find corresponding indices")

    def get_cache_path(self):
        return os.path.join(CACHE_DIR, self.name + '.pkl')

    def maybe_load_cache(self):
        path = self.get_cache_path()
        if os.path.exists(path) and False: ## XXX CACHE ALWAYS DISABLED
            try:
                with open(path, 'rb') as reader:
                    logger.info(f'Loading {self.name} from cache')
                    return pkl.load(reader)
            except Exception as e:
                logger.warning(str(e))
                pass

        return None

    def dump_to_cache(self):
        path = self.get_cache_path()
        with open(path, 'wb') as writer:
            logger.info(f'Dumping {self.name} to cache')
            pkl.dump(self.examples, writer)

    def __getitem__(self, index):
        return self.examples[index]
    
    def __len__(self):
        return len(self.examples)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
    for rel_path in XLWIC_TESET_PATHS:
        lang = rel_path.split('_')[-1]
        print('Language', lang.upper())
        data_path = os.path.join('data/xlwic_datasets/', rel_path, lang + '_test_data.txt')
        gold_path = os.path.join('data/xlwic_datasets/', rel_path, lang + '_test_gold.txt')
        XLWICDataset(data_path, 
        tokenizer, lang, 'test', 'char_2_token', 'mean', gold_path)
