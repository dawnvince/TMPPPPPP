# Code for PErMformer

# Enviroment with Conda
```bash
# Python version = 3.9.17
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

# Run Experiments
```bash
python Exp.py
```

# Results:
checkpoints and results are reserved in the directory ./cpkt. Select the results with the minimum validation loss as the final result.