# Hackathon 2022 EESTECH

<p align="center">
<img src="recordings/image.png" alt="drawing" width="300"/>
</p>
<p align="center">
<a href="#"><img alt="Last commit" src="https://img.shields.io/github/last-commit/ml-lohi/hackathon-private/main?color=green&style=flat"></a>
<a href="#"><img alt="Videos" src="https://img.shields.io/github/contributors/ml-lohi/hackathon-private?color=blue&style=flat"></a>
</p>


## What is the project?

In this cool project we work with BGT60TR13C radar chip.

We learn how to find the difference between *moving* and *static* objects, by the data that the **radar chip** gives us. When the person is classified as static we make analysis over the vital functions of the person by measure breathing as well as heart rate of the person using the cheap raw data.

## Setting up the project

The libraries needed for the project are defined in [requirements.txt](https://github.com/ml-lohi/hackathon-private/blob/main/requirements.txt)

To install needed libraries in your virtual environment run the following commands:

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

**Bug**: unfortunately the provided library `ifxdaq` only works with Windows and Linux. We tried as well Ubuntu Docker, but there comes up the issue, that the docker doesn't have an easy ability to work with usb ports of the host. This is definately a limitation (in our team 2/3 are using MacOS). Moreover it appears that the library `ifxdaq` doesn't work with the newest version of the **Python** (3.10).

## File meanings

### 1. Utils

We implemented all the needed [utilities](https://github.com/ml-lohi/hackathon-private/blob/main/utils) for working with data. 

### 2. Data overlook

Firstly we made an overlook over the data, that the **radar chip** delivers us. We measured for 30s three different data sets: static person, static person but further away from the radar and moving person. Our overlook can be found in [data_overlook.py](https://github.com/ml-lohi/hackathon-private/blob/main/data_overlook.ipynb).

### 3. Moving/Static classification
First step is to classify the objects as moving or static. 

The detailed overview is provided in the notebook [models.ipynb](https://github.com/ml-lohi/hackathon-private/blob/main/models.ipynb). In the notebook we measured for 100s each of the team membrs in static as well as in moving state. This delivers us 600s of total data for the analysis. In the notebook [linear_discriminant_ananlysis.ipynb](https://github.com/ml-lohi/hackathon-private/blob/main/linear_discriminant_ananlysis.ipynb) we can find the analysis of the data with help of linear boundry.

### 4. Vital functions analysis
For the vital functinality we analyse breathing as well as heart rate. The detailed analysis is implemented in [fourier.ipynb](https://github.com/ml-lohi/hackathon-private/blob/main/fourier.ipynb)

### 5. Final app

The final state of our project was to develop an app, which can be started, when the radar is connected to the laptop. This app is implemented in [main_app.py](https://github.com/ml-lohi/hackathon-private/blob/main/main_app.py). You can start it by simply calling 

    python main_app.py

The applicatino plots the phases every second, updates the value of the heart rate every 5s and breathing rate every 10-15s.