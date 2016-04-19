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

In the preliminary analysis that I am presenting, I used the PAMAP2 dataset from the UC Irvine Machine Learning 
repository and accessible through this URL:

https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring

I used this dataset to train a support vector machine classifier to associate the pulse readings to
intensity levels of the physical activity of the subject.
