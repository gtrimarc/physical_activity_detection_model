# Pulse2act: machine learning at work to track the intensity of physical activity!

This Jupyter notebook demonstrates the use of machine learning to detect the intensity of physical activity using the readings of wearable devices that can track body motions and basic physiological parameters such as heart rate. 

Many people have trouble motivating themselves to do vigorous exercise, either because they believe they are already active \
enough or, worse yet, they don’t know how much is enough. A smart device that can detect the level of the physical activity of an individual over time would help to keep track the stretches of intense physical activity performed. Moreover, such an exercise tracking application would allow a user to monitor the progress towars a set goal for the amount of exercise to perform in a given period of time. 

The preliminary analysis I am presenting here shows a direct correlation between the heart rate and accelleration measured by the device with intensity levels of physical activity. This model could become the engine of an app to monitor and record over time the level of physical activity of a person using a set of readings from a smartphone and/or other wearable devises that that person uses in his or her daily life. 

In this study I am using the PAMAP2 dataset from the UC Irvine Machine Learning repository to build the model. The dataset if accessible through this URL:

https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring

The dataset includes readings of a total of 52 body vitals monitored on several parts of the body of the
subjects that were studied along with with sensory data such as accelerometer, magnetometer, and gyroscope
data that precisely describe the movement in which the subjects are involved when the readings are taken.
The readings are associated to a particular type of activity in which the subjects are involved. I chose
to group these activities into four groups labeled minimum, low, medium, and high intensity activities.
The goal of the data analysis project is to apply a machine learning techniques to develop a model that
correlates sensor readings to the activity level. The final objective is to deploy the model as an app that
tracks the intensity of body activity over time using readings of vital signs and body motion from personal
smart devices.

In this preliminary study I am using a support vector machine classifier to learn the intensity of physical
activity starting from the training data. The initial tests show a promising accuracy of at least 80 %.
As next steps I plan to test other important descriptors that might be relevant to obtain an optimal model. 
I will also test other classifiers such as neural networks.
