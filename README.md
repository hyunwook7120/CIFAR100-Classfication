# **CIFAR100-Classification**
### Pytorch-CIFAR100
practice on cifar100 using pytorch

### Requirements
This is my experiment environment
1. python3.8
2. pytorch 2.4.1+cu124
3. wandb 0.18.2(optional)

## Usage
### 1. dataset
We conducted a project to classify images using the CIFAR100 dataset. The data augmentation techniques used were RandomCrop, RandomHorizontalFlip, Normalize, and CutMix. Additionally, ToTensor was used for data preprocessing.

### 2. run wandb(optional)
Install wandb
```bash
pip install wandb
```
```python
import wandb
wandb.init(project="CIFAR-100_Classification", name=config["model_name"], config=config)
```

### Git Commit Rules
| Tag Name           | Description                                               |
|--------------------|-----------------------------------------------------------|
| **Feat**           | Adds a new feature                                      |
| **Fix**            | Fixes a bug                                              |
| **!HOTFIX**        | Urgently fixes a critical bug                     |
| **!BREAKING CHANGE**| Introduces significant API changes                                |
| **Style**          | Code format changes, missing semicolons, no logic changes      |
| **Refactor**       | Refactors production code                                     |
| **Comment**        | Adds or updates necessary comments                                   |
| **Docs**           | Documentation changes                                                  |
| **Test**           | Adds or refactors test code, no changes to production code |
| **Chore**          | Updates build tasks, package manager configs, no changes to production code |
| **Rename**         | Renames or moves files or directories only         |
| **Remove**         | Removes files only                         |
