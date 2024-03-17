# Group Members
- Pau Vallespi
- Carles Pregonas
- Carlos Boned
- Marc Perez

# Code Explanation
This repository contains the code related to the project on 'Object Detection and Segmentation' of the C5: Visual Recognition of the Master in Computer Vision at UAB.

## Week 1
>To see the slides click [here](https://docs.google.com/presentation/d/1CoujhU4kM0HRyDuOHyaCCPdV8Ej7_X9nboHhPHJPP5E/edit?usp=sharing)  

<details>
  <summary>Click me to read about this week!</summary>

### Instructions
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

</details>


## Week 2

>To see the slides click [here](https://docs.google.com/presentation/d/10djQgyC_lXfmIj28mknT_SJXyfd-LtPIwiabJMvIgsA/edit#slide=id.g2bed5711158_1_7) 

<details>
  <summary>Click me to read about this week!</summary>

### Instructions

To run the evaluation process for object detection models, follow these steps:

1. **Set Up Environment:**
   - Ensure that you have all the necessary dependencies installed. You can check the requirements in the project's `requirements.txt` file.
   - Make sure you have access to the MCV server or set up the environment locally.

2. **Clone the Repository:**
   ```bash 
   git clone https://github.com/marcps8/MCV-C5-G1.git
3. **Install requirements** 
```bash
pip install -r requirements.txt
```

4. **Run the code**
  * Evaluation Script

The evaluation script evaluates object detection models using pre-trained weights on the KITTI-MOTS dataset. This involves using the evaluation.py file. Below are the instructions for running the evaluation:

Navigate to the project directory.

Run the following command in the terminal:

```bash
usage: python evaluation.py --model-index MODEL_INDEX
```
Replace MODEL_INDEX with the index of the model you want to evaluate. Choose 0 for Faster R-CNN and 1 for Mask R-CNN.
The evaluation results will be saved in the results/evaluation/{model_name} directory.

The evaluation script utilizes the inference.py and dataset.py files to perform inference and handle dataset loading, respectively.

  * Inference Script

Additionally, you can run the inference script separately to perform inference using the object detection models on new images. This involves using the inference.py file. Here's how to do it:

Navigate to the project directory.

Run the following command in the terminal:

```bash
python inference.py --model-index MODEL_INDEX --mode MODE
```

Replace MODEL_INDEX with the index of the model you want to use (0 for Faster R-CNN and 1 for Mask R-CNN), and MODE with either "training" or "testing" depending on the dataset mode.

The inference results will be saved in the results/{model_name}/{dataset_mode}/{image_name} directory for each image processed.

The inference script utilizes the inference.py file for running inference and dataset.py for handling dataset loading.

These scripts provide a comprehensive toolset for evaluating and performing inference with object detection models trained on the KITTI-MOTS dataset.
</details>

## Week 3

>To see the slides click [here](https://docs.google.com/presentation/d/1CkdWHK1STnMNZHGVYpRNxGoxr4cdO-IuxOuMx7xC8kc/edit?usp=sharing) 

<details>
  <summary>Click me to read about this week!</summary>

jaja no hay nada
</details>
