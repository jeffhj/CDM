# CDM (Generation)




## Requirements

```
fairseq==0.10.2
torch==1.7.1
tqdm==4.48.2
```



## Data

Data can be downloaded from [Google Drive](https://drive.google.com/file/d/17aFC1rdfqjkoRtR37wTir8x-XAy2h7hc/view?usp=sharing)



## Train

**Download pre-trained BART**

Download pre-trained `bart.base` or `bart.large`

```
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
```

Unzip files to `bart/bart.base/` or `bart/bart.large/`



**Preprocess data**

```
bash preprocess.sh
```



**Train model**

```
bash train.sh
```



## Generation

**Generate definitions for terms in test set**

CDM-S$5$

```
fairseq-generate input/cs-top5-bin/ --path tmp/cs-top5/checkpoint_best.pt --beam 5 --batch-size 128 --remove-bpe --no-repeat-ngram-size=5 --min-len=5 --max-len-b 100 --bpe gpt2 --gpt2-encoder-json encoder.json --gpt2-vocab-bpe vocab.bpe --scoring sacrebleu | tee output/cs-top5.out
```

CDM-S$5$,C$5$

```
fairseq-generate input/cs-top5-corr5-bin/ --path tmp/cs-top5-corr5/checkpoint_best.pt --beam 5 --batch-size 128 --remove-bpe --no-repeat-ngram-size=5 --min-len=5 --max-len-b 100 --bpe gpt2 --gpt2-encoder-json encoder.json --gpt2-vocab-bpe vocab.bpe --scoring sacrebleu | tee output/cs-top5-corr5.out
```





## Evaluation

**Extract the output**

```
grep ^D output/cs-top5.out | cut -f3- > output/cs-top5.out.sys
grep ^T output/cs-top5.out | cut -f2- > output/cs-top5.out.ref
```



**Evaluation**

```
bash RM-scorer.sh output/cs-top5.out.sys output/cs-top5.out.ref
```



## Interactive

**Run interactive mode**

```
fairseq-interactive input/cs-top5-bin  --path tmp/cs-top5/checkpoint_best.pt  --bpe gpt2  --source-lang src --target-lang tgt  --no-repeat-ngram-size 5  --beam  5  --nbest 20  --gpt2-encoder-json encoder.json  --gpt2-vocab-bpe vocab.bpe
```

