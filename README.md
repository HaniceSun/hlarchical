<p align="left">
<img src="assets/logo.png" alt="logo" width="250"/>
</p>

## Overview

**Hlarchical** is implementing deep learning models for imputing human leukocyte antigen (HLA) alleles from genotyping data. HLA genes are highly polymorphic and play a critical role in immune response, disease susceptibility, drug hypersensitivity, and transplant compatibility.

While high-resolution HLA typing using sequencing technologies is accurate, it remains expensive and unavailable for many large-scale studies. Existing methods for imputing HLA types from genotyping data include Hidden Markov Model–based SNP2HLA, machine learning–based HIBAG, convolutional neural network–based DEEP*HLA, and Transformer-based HLARIMNT, among others.

Inspired by previous studies, hlarchical aims to improve HLA imputation performance by exploring and exploiting the following features:

(1) hierarchical modeling of HLA alleles that reflects the natural structure of HLA nomenclature (e.g., 2-digit -> 4-digit resolution)

(2) mixture-of-experts (MoE) architectures that enable allele-specific experts to focus on relevant subsets of SNP features while sharing information across related alleles

(3) multi-task learning to jointly optimize predictions across multiple HLA resolutions and loci

(4) configurable model backbones (e.g., MLP, CNN, GPT) and hyperparameters to enable systematic evaluation and optimization

## Installation

- using conda

```
git clone git@github.com:HaniceSun/hlarchical.git
cd hlarchical
conda env create -f environment.yml
conda activate hlarchical
```

# Quick Start

```
hlarchical --help
hlarchical preprocess
hlarchical torch-dataset
hlarchical train
hlarchical eval
hlarchical predict
```

## Author and License

**Author:** Han Sun

**Email:** hansun@stanford.edu

**License:** [MIT License](LICENSE)
