# TH-EN Machine Translation

[Model](https://huggingface.co/wtarit/nllb-600M-th-en) | [Demo (Huggingface Space)](https://huggingface.co/spaces/wtarit/nllb-th-en-translation) | [Blog (Thai)](https://wtarit.medium.com/สร้าง-model-แปลภาษาไทย-อังกฤษ-e23cb9585f5)


# Dataset
In this project I used 2 datasets including SCB-1M and OPUS which can be downloaded from [thai2nmt project](https://github.com/vistec-AI/thai2nmt/releases/tag/scb-mt-en-th-2020%2Bmt-opus_v1.0) the data is cleaned using script in [preprocess_data](https://github.com/wtarit/th-en-machine-translation/tree/main/preprocess_data) folder. For test set I used IWSLT 2015.  

# Models
I experimented with 2 models mT5 and No Language Left Behind (NLLB). Models are evaluated using [sacrebleu](https://huggingface.co/spaces/evaluate-metric/sacrebleu) and the results are as follows.
|   Models  | Finetuning Methods |    Dataset   | num_epochs | Validation Set BLEU Score | Test Set BLEU Score | 
|:---------:|:------------------:|:------------:|:----------:|:-------------------------:|:-------------------:|
| mT5-Small |  full finetuning   |     SCB-1M   |     5      |           13.15           |        12.14        |
| NLLB-600M |  full finetuning   |     SCB-1M   |     3      |           30.74           |        21.71        |
| NLLB-600M |  full finetuning   |  SCB-1M+OPUS |     3      |           28.86           |        27.37        |
| NLLB-600M |       LoRA         |  SCB-1M+OPUS |     9      |           24.00           |        24.23        |

Note: The validation set is sampled from full dataset with 80:20 ratio for SCB-1M and 85:15 ratio for SCB-1M+OPUS.
  