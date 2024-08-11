# QMedSAM
Quantization of MedSAM / LiteMedSAM

## Environment
For inference only:
```bash
pip install opencv-python-headless openvino tqdm
```
Further install dependencies for training: install [PyTorch](https://pytorch.org/get-started/locally/), then
```
pip install brevitas timm monai pyyaml scipy
```

## Training
We use `ddp_dist_tune.py` to achieve the quantization-aware training. DDP was implemented in the style of `torch.multiprocessing.spawn`, so this script can be run as a normal python script.

The control panel of the training script is at the very beginning of the file:
```python
# control panel
CONFIG_YAML = os.path.join('config', 'finaltune_quantize_iemd-2.yaml')
PORT = 10311
GPU_RANK = [2, 7, 8, 9]
```
1. Write a configure file, in our pipeline, the three stages training protocols corresponds to `tune_quantize_ie-2.yaml`, `tune_quantize_iemd-2.yaml`, and `finaltune_quantize_iemd-2.yaml`. 
2. Specify the DDP port.
3. Specify the local GPU ids you want to use.

Before training, one should prepare the dataset. We propose `sdk.ModalityDataset` and `sdk.GatherModalityDataset` for general purposes. Organization of the dataset should be:
```
Root Dir
├── Modality Name (such as CT, MR, ...)
├── ...
    ├── Modality Dataset Name (such as AbdomenCT1K in CT)
    ├── ...
        ├── npz files
        ├── ...
```
Basically this is the same as the original dataset. We can pass the root directory to `sdk.GatherModalityDataset` and specify some of its parameters. The only parameter that worths a word is `mod_sample`, which controls the number of samples from each modality. Passing an integer or None to `mod_sample` simply spreads the strategy to all the modalities, while you can also pass a list or tuple to specify values for each modality. We provide `sdk.GatherModalityDataset` with some helpful functions, you can locate modalities in the list using `mod_names()`, fetch the real-time samples from all the modalities using `mod_samples()`, and modify `mod_sample` using `modify_mod_sample()`.

## Export
Specify the checkpoint to load and the directory to save in `export.py`. By default we will export the quantized model to `docker_python` folder, where we will build our docker.

## Inference
The general inference script is at `/docker_python/validate.py`.

## Docker
We provide two docker, one inferencing with cpp and the other running in python. You will see the python version runs faster. Thanks [team seno](https://github.com/hieplpvip/medficientsam/) for the reference in cpp code.
```bash
cd docker_python
docker build -f Dockerfile -t uestcsd .
docker save uestcsd | gzip -c > uestcsd.tar.gz
```
