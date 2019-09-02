# DLLMFS
DLLMFS stands for Deep Learning Library Made From Scratch and by scratch, I mean by array programming without importing any machine learning functions that wasn't made by me.

The project started on Matlab, since it was the only programming language I knew 3 months ago. 

After that, I learned OOP with Python and Numpy to create a class to easily compare multiple deep learning models. 

Later, when I learned about how GPUs are essential for parallel computing, I replaced every numpy arrays to torch gpu tensors. 

Right now, training a Neural Network with Cross-Entropy and Gradient Descent using my library is just as fast and accurate as Tensorflow's and Pytorch's.

**How this repo is organised.**

NeuralNetwork.py

I quickly learned that finding the best parameters of a Neural Network isn't as important as finding the best hyperparameters of the model. So the NeuralNetwork.py file contains the Class "NeuralNetwork". With that class, 1) you can create a neural network by specifying its hyperparameters (as an instance of the class), 2) you can train the model (locally or on the cloud) by inputting the training and testing data that you can get using the dataSet function in the file NNF_numpy or NNF_torch, 3) evaluate the model and 4) save the model for future uses.

NNF_numpy.py and NNF_torch.py

When you are a user who only wants to **use** this library and not **develop** it, you don't have to be concerned about these two files. Theses two files contains the exact same functions that are used to process the data, to minimise the cost function and etc. The only difference between the two, is that one uses numpy arrays and the other uses torch gpu tensors for faster training. A pycuda version will soon be uploaded. You can read the descriptions of those functions for deeper understanding.

**How to use this repo.**

I've made two files called t1.py and t2.py that shows how to use this library. However, you will notice that it uses the  sys.argv function (which it optional). If you want to train your model on the cloud, you are probably going to use cloud computing services such as AWS, GCP, Paperspace and Spell. In the t1.py, I've commented the command lines that you have input on your terminal to run the file locally or on the cloud using Spell.
