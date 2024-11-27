# SMCAnet

## Dependencies
SMCAnet was implemented on *Ubuntu 20.04* with *Python 3.8*. Before training and test, please create an environment via [Anaconda](https://www.anaconda.com/) (suppose it has been installed on your computer), and install pytorch 2.0.1, as follows,
```bash
conda create -n SMCAnet python=3.8
source activate SMCAnet
conda install torch==2.0.1
```
Besides, please install other packages using ```pip install -r requirements.txt```.

## How to test

To test the performance of U-Net, PU-Net, Baseline, and BayeSeg on the LGE CMR of MS-CMRSeg, please uncomment the corresponding line in `demo.sh`, and then run `sh demo.sh`.
```bash
# test SMCAnet
#  CUDA_VISIBLE_DEVICES=0 python -u main.py --model SMCAnet --eval --dataset ./input --sequence test_3modalities --resume ./pth/checkpoint0best6.pth --device cuda

```

## How to train
All models were trained using LGE CMR of MS-CMRSeg, and the root of training data is defined in `data/mscmr.py` as follows,
```python
root = Path('your/dataset/directory' + args.dataset)
```
Please replace `your/dataset/directory` with your own directory.

To train SMCANet, please uncomment the corresponding line in `demo.sh`, and run `sh demo.sh`.
```bash
# train SMCAnet
# CUDA_VISIBLE_DEVICES=0 python -u main.py --model SMCAnet --batch_size 8 --output_dir logs/SMCAnet --device cuda

```

