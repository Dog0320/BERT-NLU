# BERT-NLU
Pytorch implmentation for [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909).


## Requirements
```
Python3
transformers==3.1.0
torch==1.6.1+
argparse==1.4.0
seqeval==0.0.12
```

## Prepare

* Download ``google_model.bin`` from [here](https://drive.google.com/drive/folders/11i463eaaVvBrulLzSmUHdjFRgO_txBnU?usp=sharing), and save it to the ``assets/`` directory.
* Download ``dataset`` from [here](https://drive.google.com/drive/folders/1LyctA87Ve2oNb7nxovFzZbT_td9CCV24?usp=sharing), and save it to the ``data/`` directory.

### Model Training

Run example on ATIS dataset.
```
python3 train.py --data_dir data/atis/ --model_path /assets/
```
#### To use your own dataset,  modify the DataProcessor in ``data_utils.py``.

### Model Evaluation

Run example on ATIS dataset.
```
python3 evaluate.py --data_dir data/atis/ --model_path /assets/
```

### Model Prediction

Run example on ATIS dataset.
```
python3 predict.py --data_dir data/atis/ --model_path /assets/
```

### Results


|Dataset        |intent (acc) |slot (f1) |sentence|
|-------------|------------|------------|---|
|ATIS |   97.42  | 95.33     |88.80|
|Snips|     98.43 |  96.14   |91.57|

The results in this respository are not accurately tuned.

To reproduce results in the paper, you can further adjust the parameters yourself.
