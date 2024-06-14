# Efficient Quantization-Aware Training on Segment Anything Model in Medical Images and Its Deployment

This repository is the official implementation of [Efficient Quantization-Aware Training on Segment Anything Model in Medical Images and Its Deployment](). 


## Requirements
The codebase is tested with: Ubuntu 20.04 | Python 3.10 | CUDA 10.1 | Pytorch 2.0.1

Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it:

```setup
pip install -r requriements.txt
```

You can download datasets from the challenge [here](https://docs.google.com/spreadsheets/d/1QxjFs41eU6JG5KNhP576fc8MotrJ58KCrqH83HG-__E/edit?usp=sharing)
,We ues original npz files for training
## Training

Please follow the [LiteMedSAM](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM) to download its pretrained model.

To train the model(s) in the paper, you need open the ddp_training.py and edit the following variables:

```python
    GPU_RANK = [0]  # specify target gpu id
    load_fp = r'./path/to/pretrained_models.pth'
    save_path = './path_to/save_model'
    save_name = f'save_name_of_model'
    ds = sdk.GatherModalityDataset('../path/to/dataset', logger=log)
```
then 
```python
python ddp_training.py 
```
## Evaluation

To evaluate the model, you need edit validate.py:

```python
model_onnx = core.read_model(model='name_of_your.onnx')
```
then
```python
python validate.py -i /path/of/inputs -o /path/of/outputs
```

## Pre-trained Models

You can download pretrained models on the release page.