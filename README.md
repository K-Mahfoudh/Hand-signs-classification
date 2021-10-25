# Sign language translator
The aim of this project is to build a realtime hand sign translator, able to convert
 images or videos to text.
 The project is divided into two main parts, the first part is hand signs classification
based on hands images, and the second part is hand detection using Yolo V3 algorithm.
Both hand detection and signs classification are combined to create a real time sign language translator.

The idea of this project came to my mind because I wanted to learn sign language, so I thought that such a program
 could be used as a tool to teach it, or it can simply be used as a translator for communication.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required packages

```bash
pip3 install -r requirements.txt
```
If you're not placed in the root folder, use full/relative path to requirements.txt instead.

**Important**: Make sure to install the required packages, otherwise, you may 
have some package related errors (especially CUDA issues) while running the program.

## Data
Before running the program, you need to download the data first. You can find
it in [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet).

Once you download the data, place the ``/train`` and ``/test`` folders in ``/data/signs_data``  folder
, Otherwise you will need to change ``train_path`` and ``test_path`` arguments when running your code. You should have
 a similar structure: 
```
Hand-signs-classification
│     
└───data
│   │   __init__.py
│   │   data.py
│   │
│   └───signs_data
│       │
│       │───train
│       │   └───img1.png ....
│       │
│       └───test
│           └───img1.png ....      
└───....
```

## Usage
You can run the program using the following command:

```
python3 hand_sign_translator.py
```
This command will use the default parameters, if you want to change any parameter
, you will need to pass it as an argument when running the program, for example:
 ```
 python3 hand_sign_translator.py -m Test -lr 0.001
 ```
For more information about arguments, type the following command:
```
python3 hand_sign_translator.py -h
```
## Hand detection
This part is not implemented yet due to hardware issues (too much training overheated my laptop...), but it will
be implemented in the future (I hope) using a Yolo V3 algorithm.

## Contributing
All are welcome to contribute, but before making any changes, please make sure
to check the project's style. It is best if you follow the "fork-and-pull" Git workflow.

1. Fork the repo on GitHub
2. Clone the project to your own machine
3. Commit changes to your own branch
4. Push your work back up to your fork
5. Submit a Pull request.

**Important**: Do not forget to keep your cloned project up to date, and to open
a new issue in case of any major changes.

## License
[MIT](https://github.com/K-Mahfoudh/Sign-Language-Translator/blob/main/LICENSE.md)
