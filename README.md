# CLATNet: CNN-LSTM-Attention Network for Time Series Anomaly Detection

This project implements a hybrid neural network architecture (CLATNet) that leverages CNN, LSTM, Attention, and Transfer Learning for multivariate time series anomaly detection.

## Features
- Multivariate anomaly detection
- CNN + LSTM + Attention layers
- Transfer learning for reduced training time
- Preprocessing pipelines and evaluation scripts

## Folder Structure
- `notebooks/`: Example Jupyter notebooks for each dataset
- `src/`: All reusable source code
- `data/`: Raw and preprocessed datasets
- `outputs/`: Saved models and logs

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train/train.py --config configs/insdn.yaml
```

## Datasets
- SWaT
- InSDN
- Credit Card Fraud
- Gasoil Heating Loop (GHL)# CLATNet
