
# coding: utf-8

# Welcome to the first Hands On Exercise on Tensor Flow. In this exercise , you will try out some basic commands that you have learnt in the course. 
# We have created this Python Notebook with all the necessary things needed for completing this exercise. 
# You have to write your code in between the area mentioned 
# 
# `## Start Code` 
# 
# 
# 
# 
# `## End Code` 

# Importing the necessary libraries 

# In[ ]:


import tensorflow as tf
import numpy as np


# Using numpy initialize two matrix with values **[1,3,4],[3,5,3],[4,5,3]** and **[2,5,7],[3,6,8],[2,6,9]** as **matrix1** and **matrix2** respectively

# In[4]:


## Start Code 
import tensorflow as tf
import numpy as np
matrix1 = np.array([[1,3,4],[3,5,31],[4,5,3]          
                   ])
matrix2 = np.array([[2,5,7],[3,6,8],[2,6,9]
                   ])

## End Code 


# In[5]:


## Start Code 
import tensorflow as tf
import numpy as np
matrix1 = np.array([[1,3,4],[3,5,31],[4,5,3]          
                   ])
matrix2 = np.array([[2,5,7],[3,6,8],[2,6,9]
                   ])
tf_mat1 = tf.convert_to_tensor(matrix1, dtype=tf.float64) 
tf_mat2 = tf.convert_to_tensor(matrix2, dtype=tf.float64) 

## End Code 


# covert **matrix1** and **matrix2** to **tensorflow object** and call them as **tf_mat1** and **tf_mat2**, make sure to specify datatype as **"float64"**

# initilise **tensorflow variables a and b** with matrix tf_mat1 and tf_mat2 respectively.
# Also initialize all the variables before performing the operation. 

# In[6]:


## Start Code
import tensorflow as tf
import numpy as np
matrix1 = np.array([[1,3,4],[3,5,31],[4,5,3]          
                   ])
matrix2 = np.array([[2,5,7],[3,6,8],[2,6,9]
                   ])
tf_mat1 = tf.convert_to_tensor(matrix1, dtype=tf.float64) 
tf_mat2 = tf.convert_to_tensor(matrix2, dtype=tf.float64) 
a = tf_mat1
b = tf_mat2
init = np.zeros([1,3])

## End Code 


# Perform **elementwise** matrix multiplication on a, b and assign it to **final_mat1** variable 

# In[7]:


## Start Code 
import tensorflow as tf
import numpy as np
matrix1 = np.array([[1,3,4],[3,5,31],[4,5,3]          
                   ])
matrix2 = np.array([[2,5,7],[3,6,8],[2,6,9]
                   ])
tf_mat1 = tf.convert_to_tensor(matrix1, dtype=tf.float64) 
tf_mat2 = tf.convert_to_tensor(matrix2, dtype=tf.float64) 
a = tf_mat1
b = tf_mat2
init = np.zeros([1,3])
final_mat1 = np.multiply(a,b)
## End Code

determinant1 = tf.matrix_determinant(final_mat1)

 


# Perform **dot product** on **matrix a**, and **transpose of b** . 

# In[14]:


## Start Code 
import tensorflow as tf
import numpy as np
matrix1 = np.array([[1,3,4],[3,5,31],[4,5,3]          
                   ])
matrix2 = np.array([[2,5,7],[3,6,8],[2,6,9]
                   ])
tf_mat1 = tf.convert_to_tensor(matrix1, dtype=tf.float64) 
tf_mat2 = tf.convert_to_tensor(matrix2, dtype=tf.float64) 
a = tf_mat1
b = tf_mat2
init = np.zeros([1,3])
final_mat1 = np.multiply(a,b)
b_inv = tf.transpose(b)
final_mat2 = np.dot(a,b_inv)
## End Code https://2886795275-8888-host02-fresco.environments.katacoda.com/notebooks/HandOn_Basics_of_Tensorflow.ipynb#

determinant2 = tf.matrix_determinant(final_mat2)


# Execute the below piece of code. 

# In[18]:


text_file = open("Output.txt", "w")


# In[21]:


with tf.Session() as sess:
    sess.run(init)
    with open("Output.txt", "w") as text_file:
        text_file.write("determinant1 %f\n" % sess.run(determinant1))
        text_file.write("determinant2 %f" % sess.run(determinant2))    

