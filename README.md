# GMFFNet: A Gaze Estimation Method
* Gaze estimation is a vital technology for predicting the direction or specific target of an individual's gaze. We propose GMFFNet, a framework designed to regress gaze directions, based on facial image appearance. Its innovation lies in integrating multi-scale hierarchical feature interaction with an adaptive attention mechanism (CEMA), which collectively enables robust gaze estimation.

![GMFFNet](https://github.com/aname-123/GMFF-Net/blob/main/Figure/Fig1.png)

## Getting Started
### Installation
* Clone the repository
```
git clone https://github.com/aname-123/GMFF-Net.git
```
* Environment Requirements  
- Operating System: Linux (We tested on Ubuntu 24.04)  
- Python Version: 3.12 or higher  
- torch: 2.3.0  
```
pip install -r requirements.txt
```


### Datasets
* Download **MPIIFaceGaze** dataset from [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation)
* Download **Gaze360** dataset from [here](https://gaze360.csail.mit.edu/download.php)

Store the dataset to datasets/FaceBased/MPIIFaceGaze or Gaze360.
```
Datasets/
└── FaceBased/
    ├── MPIIFaceGaze/
    │   ├── Image/
    │   └── Label/
    └── Gaze360/
        ├── Image/
        └── Label/
```

We provide the code for MPIIFaceGaze dataset with leave-one-person-out evaluation.

## Acknowledgement
This work was supported by the National Key Research and Development Project of China (2024YFC2419500).

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This work is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

## Contact
For questions or feedback, feel free to reach out: P23030854097@cjlu.edu.cn.
