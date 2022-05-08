# Hackathon 2022 EESTECH
## What is the project?

In this cool project we work with BGT60TR13C radar chip.

We learn how to find the difference between *moving* and *static* objects, by the data that the **radar chip** gives us. When the person is classified as static we make analysis over the vital functions of the person by measure breathing as well as heart rate of the person using the cheap raw data.

## How we proceeded in this project?
### 1. Data overlook

Firstly we made an overlook over the data, that the **radar chip** delivers us. We measured for 30s three different data sets: static person, static person but further away from the radar and moving person. Our overlook can be found in [data_overlook.py](https://github.com/ml-lohi/hackathon-private/blob/main/data_overlook.ipynb). 

Overall the major **conclusion** that we got from there is that the phases of the moving objects are getting a lot higher than the phases of the static objects. Moreover it doesn't make difference which distance is between the objects (the only differrence is that less distanced obects are normally more noisy).

### 2. Moving/Static classification
First step is to classify the objects as moving or static. 

The detailed overview is provided in the notebook [models.ipynb](https://github.com/ml-lohi/hackathon-private/blob/main/models.ipynb). In the notebook we measured for 100s each of the team membrs in static as well as in moving state. This delivers us 600s of total data for the analysis.
#### Threshold
The easiest way for the classification, that can our [data_overlook.ipynb](https://github.com/ml-lohi/hackathon-private/blob/main/data_overlook.ipynb) delivers, is to set the threshold at some specific point.

After doing so we already get very good classification results, approximately 73% accuracy. In real time application it works even withour any further analysis good enough.

#### Simple dense neural network

Nevertheless a simple threshold setting seems to be as a shortcut in the solution. That's why we decided to feed our phases data to the neural network. 

Out of the 600s data we created 600 samples, each containing 1000 phases values. By using the network even with one neuron we would get more than 1000 learnable parameters, which is unfeasible for the amount of data that we have. That's why we decided to average the data of every second and got to the 50 values, that represent every second. This still looked approprite on the plot and the amount of learnable would be much less.

Nevertheless we got a very unstable value of accuracy. Of course the reason for this is because the data is simply not enough.

#### 1D CNN

After seeing the problem with amount of learnable parameters we decided to use a 1D CNN. CNN needs a lot less data learnable parameters, because of the sparsity.

The results were a lot better than in the dense neural network, that's why we decided to use both **Threshold** and **CNN** classification for the moving/static classification.

### 3. Vital functions analysis
For the vital functinolity we analise breathing as well as heart rate. The detailed analysis is implemented in [fourier.ipynb](https://github.com/ml-lohi/hackathon-private/blob/main/fourier.ipynb)

#### Breathing rate

To find the breathing rate we implemented FFT on the breathing data. We firstly smoothen the data with the gaussian filter, to remove the noise. After applying the low pass filter with 0.1-0.6 we apply the fft. The values 0.2-0.5 correspond to 12-30 breaths per minute, we added and subtracted 0.1 values to the boarders for additional safety. The peak of the fft is set to be the corresponding breathing rate.

#### Heart rate

The same procedure is as well used to find the heart rate, but with a lower sigma for the gaussian filter and high pass filter with boarders 0.7-4.0 (this correspods 42-240 beats per minute).

#### Peak instead FFT

We as well tried to calculate the heart and breathing rates by simple calculating of the peaks of the smoothened phases data. This turned out as well to be a solid solution.

### 4. Final app

The final state of our project was to develop an app, which can be started, when the radar is connected to the laptop. This laptop plots the phases every seconds, updates the value of the heart rate every 5s and breathing rate every 10-15s.