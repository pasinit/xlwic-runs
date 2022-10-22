# XL-WiC Models
This repository contains code and pretrained baselines (coming soon) for the [xl-wic](https://pilehvar.github.io/xlwic/) task released in this [paper](https://arxiv.org/abs/2010.06478). Results are slightly different from what was reported on the paper. Please consider these figures when comparing on XL-WiC dataset.

| | BG	|DA	|ET	|FA	|HR	|JA	|KO	|NL	|ZH	|IT	|FR	|DE	|AVG	|Epoch	|Batch Size	|EN Dev |
|:---:| :---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:	| :---: |
XLM-R Large | 62.05	|66.27	|66.15	|78.25	|64.71	|57.28	|69.92	|70.22	|61.20	|58.28	|60.20	|61.31	|64.65	|8	|32	|74.14|
XLM-R Base | |
The model averages subword embeddings to create word embeddings and concatenate the embeddings of the two target words before prediction (as also reported in the paper). 
