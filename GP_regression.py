'''
based on: Machine learning -- Introduction to Gaussian Processes, Nando de Freitas
'''
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed = 1010)
n = 5
data_x = np.linspace(3,5,n).reshape(-1,1)

def kernel(a,b):    #define kernel
    sqr_dist = np.sum(a**2,1).reshape(-1,1)+ np.sum(b**2,1) - 2*np.dot(a,b.T)
    return np.exp(-0.5*sqr_dist)

def getL(K):
    return np.linalg.cholesky(K)

mu = 0
sigma = 1
d = np.random.normal(mu,sigma,(n,1)) #generate 10 random standard normal datasets

K = kernel(data_x,data_x)             #Calculating kernel matrix for GP prior
L = getL(K+1e-6*np.eye(n))  #Calculate Cholesky decomposition         
f_prior = np.dot(L,d)       #Drawing samples from prior GP

'''Inference Step'''
N  = 50
data_x_new = np.linspace(0,10,N).reshape(-1,1)

K_ = kernel(data_x,data_x_new)
_K_ = kernel(data_x_new,data_x_new)


'''raw methode for calculating K inverse'''
InvK =  np.linalg.inv(K+1e-6*np.eye(n)) #K is singular, adding 1e-6 to make K invertable
mu_ = K_.T @ InvK @ f_prior             #Mean inference
Cov_matrix = _K_- K_.T @ InvK @ K_      #Covariance inference

# '''a better way to computer K inverse?'''
# InvK = np.linalg.inv(L.T) @ np.linalg.inv(L)
# mu_ = K_.T @ InvK @ f_prior             #Mean inference
# Cov_matrix = _K_- K_.T @ InvK @ K_      #Covariance inference


L_ = getL(Cov_matrix+1e-6*np.eye(N))        #Calculate Cholesky decomposition of infered Covariance matrix
rand_sample = np.random.normal(0,1,(N,10))   #Drawing 5 sets of random sample each of size 50
f_post = np.dot(L_,rand_sample)+mu_         #Drawing samples from posterior GP


'''Plots'''
plt.plot(data_x,f_prior,'ro',label = 'Observations')
plt.plot(data_x_new,mu_,'k-',label = 'Infered mean')
# plt.plot(data_x_new,mu_+1.65*np.diag(Cov_matrix).reshape(-1,1), 'g--',label='CI')
# plt.plot(data_x_new,mu_-1.65*np.diag(Cov_matrix).reshape(-1,1), 'g--')
plt.fill_between(data_x_new.reshape(-1),(mu_+1.65*np.diag(Cov_matrix).reshape(-1,1)).reshape(-1),(mu_-1.65*np.diag(Cov_matrix).reshape(-1,1)).reshape(-1),facecolor = 'grey',alpha = 0.5, label = 'C.I.')
plt.plot(data_x_new,f_post,'-')
plt.legend()
plt.show()