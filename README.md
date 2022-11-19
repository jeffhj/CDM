# README

The code and data for "[Understanding Jargon: Combining Extraction and Generation for Definition Modeling](https://arxiv.org/pdf/2111.07267.pdf)" (EMNLP '22)

## Introduction

We propose to combine extraction and generation for jargon definition modeling: first extract self- and correlative definitional information of target jargon from the Web and then generate the final definitions by incorporating the extracted definitional information. Our framework is remarkably simple but effective: experiments demonstrate our method can generate high-quality definitions for jargon and outperform state-of-the-art models significantly, e.g., BLEU score from 8.76 to 22.66 and human-annotated score from 2.34 to 4.04.

![image](https://user-images.githubusercontent.com/47152740/202871630-5afbafbb-49dc-48f8-8ba0-5af716a79c46.png)


## Usage

Please refer to the detailed `README.md` in `./extraction/` and `./generation/`



## Data

Data can be downloaded from [Google Drive](https://drive.google.com/file/d/17aFC1rdfqjkoRtR37wTir8x-XAy2h7hc/view?usp=sharing)



## Generated definitions

Stored in `./sample/generated_definition_for_cs_term.txt`



## Citation

The details of this repo are described in the following paper. If you find this repo useful, please kindly cite it:

```
@inproceedings{huang2022understanding,
  title={Understanding Jargon: Combining Extraction and Generation for Definition Modeling},
  author={Huang, Jie and Shao, Hanyin and Chang, Kevin Chen-Chuan and Xiong, Jinjun and Hwu, Wen-mei},
  booktitle={Proceedings of EMNLP},
  year={2022}
}
```
