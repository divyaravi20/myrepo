
# coding: utf-8

# Welcome to the second hands-on exercise for Tensorflow. In this exercise , you will be building a linear regression model using tensorflow. 
# You will be given a vector of input and output data, you need to find the linear relationship between these two data which is in the form  $$y = w*x + b$$

# Execute the code in each cell using the **Shift + Enter** command

# Importing the necessary libraries.  Import for cherry pick

# In[6]:


import numpy as np
import tensorflow as tf
import numpy.random as rand



# Using numpy initialise training input data 
# **trainX** with values [4.4,7.2,3.712,6.42,4.168,8.79,7.88,7.59,2.167,7.042,10.71,5.33,9.97,5.64,9.27,3.1,3.9]
# 
# **trainY** with values [2.28644, 3.25412, 2.0486672, 2.984552, 2.2062608, 3.803624 ,3.489128, 3.388904 ,1.5147152, 3.1995152, 4.467176, 2.607848 ,4.211432, 2.714984, 3.969512, 1.83716, 2.11364]

# In[7]: testing with a new cherry pick 2 3 and finally. Test cherry pick



## Start Code - test1
trainX = np.array([4.4,7.2,3.712,6.42,4.168,8.79,7.88,7.59,2.167,7.042,10.71,5.33,9.97,5.64,9.27,3.1,3.9])
                         
trainY = np.array([4.4,7.2,3.712,6.42,4.168,8.79,7.88,7.59,2.167,7.042,10.71,5.33,9.97,5.64,9.27,3.1,3.9])
## End Code 

num_samples = trainX.shape[0]


# Define placeholders for input and output data as **X** and **Y** . Let them be of data type **float32** 

# In[8]:


X = tf.placeholder("float")
Y =tf.placeholder("float")


# Since this is a **linear regression** problem the output will be in the form of **w * trainX + b** 
# 
# Initialse **w** and **b** with random values using **rand.randn()** function. Make sure you declare them as **tensorflow variables**

# In[9]:


## Start Code 
w = rand.randn()
b = rand.randn()
## End Code 


# implement **X * W + b** using appropriate tensorflow functions to predict the output **Y**.
# 
# Train the model using gradient descent to minimize the cost. 
# Initialize the number of iterations and learning rate 
# 
# **TIP:** start with small learning rate and large iteration.

# In[ ]:


## Start Code 

num_iter = 50
learning_rate = 0.1


pred = 
cost = 

optimizer = 
train = optimizer.minimize(cost)


## End Code 


# 
# Execute the below piece of code. 

# In[ ]:


text_file = open("Output.txt", "w")
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    for i in range(num_iter):
        session.run(train, feed_dict={X: trainX , Y: trainY})
    w = session.run(w)
    b = session.run(b)
    with open("Output.txt", "w") as text_file:
        text_file.write("w= %f\n" % w)
        text_file.write("b= %f" % b)

