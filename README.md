# OpenCV HoG Classifier Trainer
This repository includes scripts to make training HoG feature classifier on OpenCV easier.

### Requirements
* Python
* libsvm
* numpy
* OpenCV (must be exposed to Python interface)

### Usage
Change necessary variables in train.sh and execute the file.

### Variables
| Variable      | Description                          |
| ------------- | -------------                        |
| POS_PATH      | Path to positive training set images |
| NEG_PATH      | Path to negative training set images |
| OUTPUT_PATH   | Path to save resulting files         |
| MODEL_NAME    | Name to save the resulting files     |
| WIDTH         | Sliding window width                 |
| HEIGHT        | Sliding window height                |
| TRAINER_PATH  | Path to this folder                  |

### Output files information
| Filename           | Description                                              |
| -------------      | -------------                                            |
| modelname          | File containing extracted HoG features, in libsvm format |
| modelname.model    | File containing results form libsvm format               |
| modelname.features | File containing hyperparameter detecting vectors         |
modelname.features is the only required file in runtime.

### Sliding window size information
Currently, the width/height value must follow the following rule:
```
(WIDTH - 16) % 8 = 0
(HEIGHT - 16) % 8 = 0
```
 This will be changed in the future.
