# Continuous Meta-Learning without Tasks

This code accompanies the paper **Continuous Meta-Learning without Tasks** by James Harrison, Apoorva Sharma, Chelsea Finn, and Marco Pavone. 

This repo also contains pytorch implementations of [ALPaCA](https://arxiv.org/abs/1807.08912) and **PCOC** (introduced in the above paper), simple Bayesian meta-learning algorithms for regression and classification respectively. 

## To install:

First, install the MOCA package via

```
pip install -e .
```

then install dependencies via 

```
pip install -r requirements.txt
```

## To train:

Simple sinusoid experiment:

```
python experiments/train.py --train.experiment_id=0 --train.seed=0 --train.experiment_name='example'
```

Note that GPU usage is disabled by default. To enable, add argument:

```
--data.cuda=1
```

where the argument corresponds to the device. 

## To test:

```
python experiments/test.py --model.model_name='7500.pt' --train.experiment_id=0 --train.train_experiment_name='example' --train.experiment_name='example' 
```


