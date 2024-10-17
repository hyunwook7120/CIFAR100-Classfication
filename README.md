# **CIFAR100-Classification**
### Pytorch-CIFAR100
practice on cifar100 using pytorch

### Requirements
This is our experiment environment
1. python3.8
2. pytorch 2.4.1+cu124
3. wandb 0.18.2(optional)

If your virtual environment meets upper conditions, you can use
```bash
pip install -r requirements.txt
```
But you should be careful your CUDA Version is same with ours.

### Restrictions
- 50,000 training data
- No outsource
- No test time adaptation
- Training time (~24 time)
- Can use single gpu

## Table of Usage
- [Dataset](#Dataset)
- [Data Preprocessing](#Data-preprocessing)
- [Data Augmentation](#Data-augmentation)
- [Run wandb(optional)](#Run-wandb-optional)
- [Train the model](#Train-the-model)
- [Results](#Results)
- [Our Best Model](#Our-best-model)
- [Utility](#Utility)
- [Others](#Others)


## Dataset
### 1.Dataset
We conducted a project to classify images using the CIFAR100 dataset.    

## Data-preprocessing
### 2. Data Preprocessing
- To Tensor()
- Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
- CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
- CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

## Data-augmentation
### 3. Data Augmentation
- RandomCrop(32, padding=4)
- RandomHorizontalFlip()
- CutMix

## Run-wandb-optional
### 4. Run wandb(optional)
Install wandb
```bash
pip install wandb
```
```python
import wandb
wandb.init(project="CIFAR-100_Classification", name=config["model_name"], config=config)
```
If you want to disable wandb:
```python
os.environ[“WANDB_DISABLED”] = 'true'
```

## Train-the-model
### 5. Train the model
You can use two baseline code:
- baseline.ipynb (Cutmix X)
- baseline_cutmix.ipynb (Cutmix O) <- **for our best model**

If you use validation data for per-epoch training:
- You can check the current best model's test accuracy using test.ipynb while simultaneously training the model.
elif you use test data for per-epoch training:
- All per-epoch accuracy is test accuracy

## Results
### 6. Results
| Model                  | Scheduler            | Optimizer | Augmentation | Memory       | Training Samples | Epochs | Best Epochs |Top 1 Acc | Top 5 Acc | Super Acc | Runtime |
|------------------------|----------------------|-----------|--------------|--------------|-------------|--------|--------------|-----------|-------------|-----------|-------------|
| ResNet_18              | MultiStepLR          | NAG  | cutmix       | pin_memory    | 50000       | 250    | 240    | 81.36     | 95.59     | 88.98     | 4h 11m 37s |
| Wide-ResNet_28-10      | ReduceLROnPlateau     | NAG  | cutmix       | pin_memory    | 50000       | 300    | - | 82.22     | 96.12     | 89.81     | 1d 1h 51m |
| ResNeXt_101            | ReduceLROnPlateau     | NAG  | cutmix       | pin_memory    | 40000       | 250    | - | 80.33     | 95.65     | 88.78     | 6h 6m 43s |
| PyramidNet             | ReduceLROnPlateau     | NAG  | cutmix       | pin_memory    | 50000       | 300    | - | 82.28     | 96.52     | 90.04     | 1d 15h 51m |
| PyramidNet_ShakeDrop   | ReduceLROnPlateau     | NAG  | cutmix       | pin_memory    | 50000       | 300    | 210    | 84.77     | 97.28     | 91.72     | 23h 27m 41s|   

At this time, "Runtime" refers to the total time, including model training time, evaluation time, and data analysis time.

## Our-best-model
### 7. Our Best Model
We conducted model ensemble by combining Shakedrop + PyramidNet, which had the highest accuracy, and ResNet18 with CutMix, which had relatively high accuracy and a shorter runtime. 

- Model : ResNet18 + PyramidNet_Shakedrop
- Parameters of ResNet_18 :
``` python
  {
  batch_size : 128,
  num_epochs : 240,
  lr : 0.1,
  momentum : 0.9,
  weight_decay : 5e-4,
  nesterov : True,
  gamma(factor) : 0.2,
  warm : 1,
  plateau_patience : 15,
  pin_memory : True,
  depth : 110,
  alpha : 270,
  beta : 1.0,
    
  resume : False,
  scheduler : ReduceLROnPlateau,
  train_50000 : True
  }
```
- Parameters of PyramidNet_Shakedrop :
``` python
  {
  batch_size : 128,
  num_epochs : 210,
  lr : 0.1,
  momentum : 0.9,
  weight_decay : 5e-4,
  nesterov : True,
  gamma(factor) : 0.2,
  warm : 1,
  plateau_patience : 15,
  pin_memory : True,
  depth : 110,
  alpha : 270,
  beta : 1.0,
    
  resume : False,
  scheduler : ReduceLROnPlateau,
  train_50000 : True
  }
```
The final Top-1 accuracy, Top-5 accuracy, and Super-Class accuracy are as follows.
| Top-1 Accuracy | Top-5 Accuracy | Super Class Accuracy | Total Accuracy |
|----------------|----------------|----------------------|----------------|
|      85.31     |      97.47     |         91.94        |      274.72    |


## Utility
### 8. Utility
This implements utils.py:
- EarlyStopping
- WarmUpLR

If you set "resume"=True or execute Best Model Test:
- best_acc_weights
- last_epoch
- most_recent_folder
- most_recent_weights

p.s Resume is used when continuing model training from a saved checkpoint.
Be careful when using the ReduceLROnScheduler, as the patience will be reset.

## Others
### 9. Others
(1) If you set tran_50000 == False:
You can use two version of data splitting
- RandomSampler
- Stratified (if Stratified_data == True)

(2) By default, warmup scheduler is applied.
If you want to disable warmup scheduler, set "warm" = 0

(3) If you want to use MultiStepLR Scheduler, set "scheduler" = "MultiStepLR".
The learning rate will decrease by gamma when the epoch reaches the milestones.


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
