# Learning to Learn without Forgetting using Attention

This repository is the official implementation of [*Learning to Learn without Forgetting using Attention*](https://arxiv.org/pdf/2408.03219) presented at CoLLAs 2024.

## TL;DR
We propose meta-learning a transformer-based optimizer to enhance continual learning. This meta-learned optimizer uses attention to learn the complex relationships between model parameters across a stream of tasks, and is designed to generate effective weight updates for the current task while preventing catastrophic forgetting on previously encountered tasks. 

## Available datasets

SplitMNIST.

RotatedMNIST.

SplitCIFAR-100.

## Usage

To run the code, use *main_splitmnist.py*, *main_rotatedmnist.py*, *main_cifar100.py*.

These codes will train the model and compute the average accuracy, the backward transfer (BWT), and the forward transfer (FWT) metrics.

## Notes
This code has been tested with Python 3.10.12 and PyTorch 1.13.1 with CUDA 11.7.

## Cite as
```
@inproceedings{vettoruzzo2024learning,
  title={Learning to Learn without Forgetting using Attention},
  author={Vettoruzzo, Anna and Joaquin, Vanschoren and Mohamed-Rafik, Bouguelia and Thorsteinn, R{\"o}gnvaldsson},
  booktitle={Conference on Lifelong Learning Agents (CoLLAs), 2024},
  year={2024}
}
```
