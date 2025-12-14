# Aip114 final
## Environment
```
conda create -n aip114 python==3.12
conda activate aip114
pip install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Dataset
Extract `lunch500.zip` here.</br>
The directory should look like:
```
.
├── README.md
├── config.py
├── dataset.py
└── lunch500
    ├── train
    └── valid
...
```

## Train
```
python ./train.py
```
Use `--model ResNet18/ MobileNet_V3_Small` to specify model, `--flip` to enable RandomHorizontalFlip(), `--cutmix` to enable CutMix.
</br>
e.g.
```
python3 ./train.py --model ResNet18 --flip
```
Will train ResNet18 with RandomHorizontalFlip().

Edit `config.py` to modify the hyperparameters.

## Test
```
python ./test.py
```
Use `--model ResNet18/ MobileNet_V3_Small` to specify model, `--flip`, `--cutmix` to specify data argumentation it uses.
</br>
e.g.
```
python3 ./test.py --model MobileNet_V3_Small --flip --cutmix
```
Will test MobileNetV3-Small trained using RandomHorizontalFlip() and CutMix on a valid set.

