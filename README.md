# RNNIK-MKF
This is the repo for RNNIK-MKF human arm motion prediction.

## Dependencies
The repo requires matlab to be installed with python usage. Please follow the [setup guide](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) to setup matlab for python.

## Quick Start
### Train RNN Model
The wrist predictor requires a pretrained RNN model. Adjust the parameters in `params.yaml`. Setup the training data to be used for training in `train.py`.

To start training
```bash
$ python3 train.py
```

### RNNIK-MKF
Adjust the parameters in `params.yaml`. Setup the test data in `RNNIK_MKF.py`.

To run RNNIK-MKF:
```bash
$ python3 RNNIK_MKF.py
```

### MKF Performance Investigation
To investigate the parameter changing
```bash
$ # Turn on MKF online adaptation
$ python3 RNNIK_MKF.py
$ python3 plot_params.py
```
To investigate the adaptation error
```bash
$ # Turn on MKF online adaptation
$ python3 RNNIK_MKF.py
$ # Turn off MKF online adaptation
$ python3 RNNIK_MKF.py
$ python3 plot_params.py
```

## Parameters
`params.yaml` contains all the parameters used in training and testing for RNNIK-MKF.

* `prediction_step`: The prediction horizon of RNNIK-MKF.
* `arm_predictor`:
  * `adapt`: Turns on/off the MKF online adaptation for arm predictor.
  * `lambda`: The forgetting factor. Range in [0, 1].
  * `EMA_v`: EMA smoothing on parameter update step. Range in [0, 1].
  * `EMA_p`: EMA smoothing on covariance matrix. Range in [0, 1].
* `wrist_predictor`:
  * `adapt`: Turns on/off the MKF online adaptation for wrist predictor.
  * `lambda`: The forgetting factor. Range in [0, 1].
  * `EMA_v`: EMA smoothing on parameter update step. Range in [0, 1].
  * `EMA_p`: EMA smoothing on covariance matrix. Range in [0, 1].
  * `enable_multi_epoch`: Turns on/off the multi-epoch update strategy.
  * `multi_epoch_epsilon1`: Tune the value if multi-epoch update is enabled.
  * `multi_epoch_epsilon2`: Tune the value if multi-epoch update is enabled.
  * `model_path`: The path to the pre-trained RNN model.
  * `input_seq_len`: The horizon for historical trajectory.
  * `input_size`: The dimension of the input.
  * `hidden_size`: The size of the hidden layer.
  * `num_layers`: The number of layers of RNN.
  * `output_size`: The dimension of the output.
* `Train_wrist_predictor`:
  * `train`: `True` if want to train a model. `False` if want to skip training and evaluate a model specified in `model_path`.
  * `save_model`: If `True`, the program will automatically save the trained checkpoint models to the path specified in `output_folder`. If `False`, no model will be saved.
  * `output_folder`: The directory to save all trained checkpoint models.
  * `model_path`: Only matters when `train` is set to `False`. The path to the model that want to evaluate.
  * `learning_rate`: The learning rate for training the model.
  * `num_epochs`: The number of epochs to train the model.
