## Implementation of linear regression using gradient descent
import numpy as np

class linear_regression:
    def __init__(self, learning_rate, iterations, 
               fit_intercept=True, normalize=False, coef=None):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coef = coef
        self.cost =0
  
    #Normalizing the X values by subracting it with the mean and dividing it with standard deviation.
    def normalize_Data(self,X):
        no_of_features=X.shape[1]
        X_normalized=X
    
        Mean=np.zeros(no_of_features)
        Standard_deviation=np.zeros(no_of_features)
        
        X_normalized= X-np.mean(X,axis=0)/np.std(X,axis=0)
        return X_normalized 


    def fit(self, X, y):
       
    #If we put normalize as true then only it will normalize the X values 
        if self.normalize:
            X=self.normalize_Data(X)
            
    #Taking into account the number of columns and rows of the data and also the lenght of them.
        No_of_columns=X.shape[1]
        no_of_rows=X.shape[0]
        length_of_X= len(X)
        length_of_y= len(y)
       
    #If we put intercept to true then add one's column wise; c_ adds values column wise 
    #if we put intercept is false then the X remains the same
        if self.fit_intercept:
            Weights_dimension=No_of_columns + 1
            Modified_X= np.c_[np.ones((length_of_X,1)),X]
        else:
            Weights_dimension=No_of_columns
            Modified_X = X
            
        #M is the weight vector. We are initializing the weight vector by taking zeros. We can also take random values to intialize it.
        self.M=np.zeros(Weights_dimension)
        X_T=np.transpose(Modified_X)
        self.cost=0

        for i in range(self.iterations):
            #y_hat is the y we predicted which can be obtained by multipying our X and M which is our weight vector
            y_hat=np.dot(Modified_X,self.M)
            error_vector= np.dot(X_T,y_hat-y)
            #implementing the actual formulae weight=weight-(1/number of rows)*learning rate*(summ of ypredicted-actualy)*X
            #the error is the summation of ypred-y actual 
            self.M=self.M-(1/no_of_rows)*self.learning_rate*(error_vector)
            self.cost= np.sum((y_hat-y)**2)/2*no_of_rows
        return self.M,self.cost


    def predict(self, X):
        length_of_X= len(X)
    
        if self.fit_intercept:
            Modified_X= np.c_[np.ones((length_of_X,1)),X]
        else:
            Modified_X = X
            
            #The predicted value 
            prediction=np.dot(X,self.M)
        return prediction


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

#loading the boston data set 
dataset = load_boston()
X = dataset.data
y = dataset.target

#Splitting the data set into 70percent train and 30percent test 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#Calling the linear regression class which i have designed above
regresser=linear_regression(learning_rate=0.000001,iterations=5000,fit_intercept=False, normalize=True, coef=None)

#fitting the model 
regresser.fit(X_train,y_train)

#predicted values 
y_pred=regresser.predict(X_test)

y_pred