# status
**still refactoring**

# purpose
predict ostia sst  using tensorflow conv-lstm
partially recoverd
need additional fix.

# download
- conda env create --prefix venv --file environment.yml
install dependencies
in prep folder, run script by order.

# train
- python train_tensorflow/train.py

# validate
- python train_tensorflow/validate.py

# torch train
- conda env create --prefix venv --file torch_environment.yml

# todo
- fix ftp download
- fix hydra config
- fix input, output path
