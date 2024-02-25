# Group Members
- Pau Vallespi
- Carles Pregonas
- Carlos Boned
- Marc Perez

# Code Explanation
This repository contains the code related to the project on 'Object Detection and Segmentation' of the C5: Visual Recognition of the Master in Computer Vision at UAB.


## Instructions
To run the training process, follow these steps:

1. **Set Up Environment:**
   - Make sure you have all the necessary dependencies installed. You can check the requirements in the project's `requirements.txt` file.
   - Ensure that you have access to the MCV server or set up the environment locally.

2. **Clone the Repository:**
```bash 
git clone https://github.com/marcps8/MCV-C5-G1.git
```

3. **Install requirements** 
```bash
pip install -r requirements.txt
```

4. **Run the code**
```bash
usage: model_normal.py [-h] [--load-model] [--save-plots]

optional arguments:
  -h, --help    show this help message and exit
  --load-model  Loads model from path specified in code.
  --save-plots  Stores plots to paths specified in code.
```
There are two arguments that can be used, `load-model` and `save-plots`. The first one loads the model from a specified path in the code whereas the second stores the plots in `jpg` files to the paths in the code.
