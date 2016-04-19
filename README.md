# Pulse2act : a model that turns pulse readings into intensity of physical activity

IPython notebook to develop a model that detects levels of physical activities using pulse and temperature 
readings from a smartwatch. 

Millions of Americans are obese or overweight in part due to a lack of vigorous exercise. Many people have trouble 
motivating themselves,  either because they believe they are already active enough or, worse yet, they don’t know 
how much is enough. The preliminary data analysis I am presenting shows a direct correlation between body temperature 
and heart rate measured at the hand. This is important, because this data and the model that I propose to develop 
in this project could be used to design wearable devices that track the intensity of the physical activity using simple
and inexpensive temperature and pulse sensors. This project has the potential to lead to the development of a wearable 
activity monitoring device similar to the Apple watch, but more cost effective.

Here, I include the IPython notebook for a prelimirary study of the developmenr of a model that detects levels of 
physical activities using pulse and temperature readings for instance from a smartwatch. I am using the PAMAP2 dataset 
from the UC Irvine Machine Learning repository to build the model. The dataset if accessible through this URL:

https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring

The dataset includes readings of a total of 52 body vitals monitored on several parts of the body of the
subjects that were studied along with with sensory data such as accelerometer, magnetometer, and gyroscope
data that precisely describe the movement in which the subjects are involved when the readings are taken.
The readings are associated to a particular type of activity in which the subjects are involved. I chose
to group these activities into four groups labeled minimum, low, medium, and high intensity activities.
The goal of the data analysis project is to apply a machine learning techniques to develop a model that
correlates sensor readings to the activity level. The final objective is to deploy this model into a smart
wearable device, for instance a smart watch to recognize the intensity of body activity tracking over time 
readings of vital signs such as body temperature and pulse as well as other sensor data available from the device.

In this preliminary study I trained a support vector machine classifier to associate the pulse and temperature
readings at the wrist to intensity levels of the physical activity of the subject obtaining a model with an 
accuracy of 64.8 %. In this project I plan to test other important descriptors that might be relevant to
obtain an optimal model, and I will test other classifier models including neural networks or other deep learning
techniques.
