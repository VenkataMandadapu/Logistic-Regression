import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt('hw2_data.txt', delimiter=',')

train_X = data[:, 0:2]
train_y = data[:, 2]

m_samples, n_features = train_X.shape
print ("# of training examples = ", m_samples)
print ("# of features = ", n_features)


#Sigmoid Function
def sigmoid(z):
    h = z.shape
    s = np.zeros(h)
    
    for i in range(h[0]):
        k = 1/(1+np.exp(-z[i]))
        s[i] = k
    return s


z = np.array([[1, 2], [-1, -2]])
f = sigmoid(z)
print (f)

#Cost Function
def cost_function(theta, X, y):

    cost = -((np.sum((y * np.log(sigmoid(np.dot(X,theta)))) + ((1-y) * np.log(1 - sigmoid(np.dot(X,theta))))))/(X.shape)[0])
    
    return cost

#Gradient Computation

def gradient_update(theta, X, y):
    
    h = sigmoid(np.dot(X,theta))
    grad = np.dot(X.transpose(),(h-y))
    
    grad = grad / (X.shape[0])  
    
    return grad


#Gradient Checking

def gradient_check(theta, X, y):

    g = (cost_function(theta,X+np.array([10**-4]),y) - cost_function(theta,X-np.array([10**-4]),y))/2*(10**-4)
    
    return g


# Gradient Descent and Decision Boundary

def gradient_descent(theta, X, y, alpha=1e-2, max_iterations=400):
 
    alpha *= X.shape[0]
    iteration = 0
    
    
    
    global maximum
    scaledX = np.copy(X)
    maximum = np.max(abs(scaledX))
    scaledX /= maximum
    scaledX = np.insert(scaledX,0,1,axis = 1)
    
    
    
    while(iteration < max_iterations):
        iteration += 1
        
        gradient = gradient_update(theta,scaledX,y)
        theta = theta - alpha * gradient
    
        if iteration % 25 == 0 or iteration == 1:
            cost = 0
    
    
            cost = cost_function(theta,scaledX,y)

            
            print ("[ Iteration", iteration, "]", "cost =", cost)
            plt.rcParams['figure.figsize'] = (5, 4)
            plt.xlim([20,110])
            plt.ylim([20,110])
            
            pos = np.where(y == 1)
            neg = np.where(y == 0)
            plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
            plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
            plt.xlabel('Exam 1 score')
            plt.ylabel('Exam 2 score')
            plt.legend(['Admitted', 'Not Admitted'])
            t = np.arange(10, 100, 0.1)
            
            
            h = np.copy(t)
            h /= maximum
            f = -(theta[0] + np.dot(theta[1],h))/theta[2]
            f *= maximum
            plt.plot(t,f)
            
            plt.show()
               
    return theta


initial_theta = np.random.randn(train_X.shape[1]+1)        
    

learned_theta = gradient_descent(initial_theta, train_X, train_y)


#Predicting

def predict(theta, X):
    global maximum
    X /= maximum
    X = np.insert(X,0,1,axis=1)
    predicted_labels = np.zeros((X.shape[0],1))
    probabilities = sigmoid(np.dot(X,theta))
    for i in range(probabilities.shape[0]):
        if probabilities[i] > 0.5:
            predicted_labels[i,0] = 1
        else:
            predicted_labels[i,0] = 0
    
    predicted_labels = predicted_labels.transpose()
    return probabilities, 1*predicted_labels 

t_prob, t_label = predict(learned_theta, np.copy(train_X))
t_precision = t_label[np.where(t_label == train_y)].size / float(train_y.size) * 100
print ("=== For autograder ===")
print('Accuracy on the training set: %s%%' % round(t_precision,2))
