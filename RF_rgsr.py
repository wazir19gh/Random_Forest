# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 15:09:44 2021

@author: Wazir
"""

import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor

class Random_Forest_Regressor:
    def __init__(self,x,y,n,col_samp = True):
        self.x = x
        self.y = y
        self.n = n
        self.col_samp = col_samp
        
    def bootstrapping(self):
        ''' This function would generate a bootstrap sample from the given data'''
        # Selecting the 60% of row indices without replacement
        selected_rows = list(np.random.choice(np.arange(len(self.x)), size = int(0.60*(len(self.x))), replace = False))
        # Selecting the remaining 40% from the above 60% of the indices
        repeated_rows = list(np.random.choice(np.array(selected_rows), size = len(self.x)-len(selected_rows)))
        
        
        if self.col_samp:
            # Selecting the column indices anywhere between 3 and the number of features in the dataset
            selected_columns = list(np.random.choice(np.arange(self.x.shape[1]), size = np.random.choice(np.arange(3,self.x.shape[1])), replace = False))
        
            # Selecting the non-repeating data from the original data
            sampled_data = self.x[selected_rows,:][:,selected_columns]
            initial_sampled_target_data = self.y[selected_rows]
            
            # Selecting the repeated data
            replicated_sampled_data = self.x[repeated_rows,:][:,selected_columns]
            replicated_sampled_target_data = self.y[repeated_rows]
            
            # Concatenating the non-repeating and repeated data
            sampled_input_data = np.vstack((sampled_data,replicated_sampled_data))
            sampled_target_data = np.vstack((initial_sampled_target_data.reshape(-1,1),replicated_sampled_target_data.reshape(-1,1)))
            
            return list(sampled_input_data),list(sampled_target_data),selected_rows,selected_columns
        
        else:
            # Selecting the non-repeating data from the original data
            sampled_data = self.x[selected_rows,:]
            initial_sampled_target_data = self.y[selected_rows]
            
            # Selecting the repeated data
            replicated_sampled_data = self.x[repeated_rows,:]
            replicated_sampled_target_data = self.y[repeated_rows]
            
            # Concatenating the non-repeating and repeated data
            sampled_input_data = np.vstack((sampled_data,replicated_sampled_data))
            sampled_target_data = np.vstack((initial_sampled_target_data.reshape(-1,1),replicated_sampled_target_data.reshape(-1,1)))
            
            return list(sampled_input_data),list(sampled_target_data),selected_rows
            
    def train(self):
        # Initializing four different lists to store the various bootstrapped samples 
        # and their corresponding row and column indices
        if self.col_samp:
            self.list_input_data =[]
            self.list_output_data =[]
            self.list_selected_row= []
            self.list_selected_columns=[]
            
            for i in range(self.n):
                a,b,c,d = self.bootstrapping()
                self.list_input_data.append(a)
                self.list_output_data.append(b)
                self.list_selected_row.append(c)
                self.list_selected_columns.append(d)
                
            # Initializing a list to store the various trained Decision Tree which is the Random Forest
            self.mod_list = []
            for i in range(len(self.list_input_data)):
                DT = DecisionTreeRegressor(max_depth = None)
                DT.fit(self.list_input_data[i],self.list_output_data[i])
                self.mod_list.append(DT)
            
            # Initializing a list to store the predictions of the n base learners.
            y_pred_list = []
            for i in range(len(self.mod_list)):
                Y_pred = self.mod_list[i].predict(self.x[:,self.list_selected_columns[i]]).reshape(-1,1)
                y_pred_list.append(Y_pred)
            
            # Converting the list of predictions to an array with each element of the array being a vector of predictions of n base learners
            y_pred_arr = np.array(y_pred_list)
            
            # Getting the median predictions of each of the data point
            fin_y_pred = np.median(y_pred_arr,axis = 0)
            
            # Getting the training mse
            train_mse = mse(self.y,fin_y_pred)
            
            # Printing the training mean squared error
            print("The mean squared error on the training data of the random forest regressor is : {}".format(np.round(train_mse,4)))
        
        else:
            self.list_input_data =[]
            self.list_output_data =[]
            self.list_selected_row= []
                        
            for i in range(self.n):
                a,b,c = self.bootstrapping()
                self.list_input_data.append(a)
                self.list_output_data.append(b)
                self.list_selected_row.append(c)
                                
            # Initializing a list to store the various trained Decision Tree which is the Random Forest
            self.mod_list = []
            for i in range(len(self.list_input_data)):
                DT = DecisionTreeRegressor(max_depth = None)
                DT.fit(self.list_input_data[i],self.list_output_data[i])
                self.mod_list.append(DT)
            
            # Initializing a list to store the predictions of the n base learners.
            y_pred_list = []
            for i in range(len(self.mod_list)):
                Y_pred = self.mod_list[i].predict(self.x).reshape(-1,1)
                y_pred_list.append(Y_pred)
            
            # Converting the list of predictions to an array with each element of the array being a vector of predictions of n base learners
            y_pred_arr = np.array(y_pred_list)
            
            # Getting the median predictions of each of the data point
            fin_y_pred = np.median(y_pred_arr,axis = 0)
            
            # Getting the training mse
            train_mse = mse(self.y,fin_y_pred)
            
            # Printing the training mean squared error
            print("The mean squared error on the training data of the random forest regressor is : {}".format(np.round(train_mse,4)))
            
    def oob_score(self):        
        y_pred_oob = []
        oob_score_dict = {}
        
        if self.col_samp:
            for i in range(len(self.list_selected_row)):
                oob_indices = list(set(range(len(self.x))) - set(self.list_selected_row[i]))
                for index in oob_indices:
                    Y_pred = self.mod_list[i].predict(self.x[index,self.list_selected_columns[i]].reshape(1,-1))
                    if index in oob_score_dict.keys():
                        oob_score_dict[index].append(Y_pred)
                    else:
                        oob_score_dict[index] = [Y_pred]
        
            sorted_oob_score_list = sorted(oob_score_dict.items(), key = lambda items:int(items[0]))
        
            for tup in sorted_oob_score_list:
                y_pred_oob.append(np.median(tup[1]))
                
            oob_score = mse(self.y,np.array(y_pred_oob))            
            print("The oob score for the random forest regressor on the training data is : {}".format(np.round(oob_score,4)))
        else:
            for i in range(len(self.list_selected_row)):
                oob_indices = list(set(range(len(self.x))) - set(self.list_selected_row[i]))
                for index in oob_indices:
                    Y_pred = self.mod_list[i].predict(self.x[index].reshape(1,-1))
                    if index in oob_score_dict.keys():
                        oob_score_dict[index].append(Y_pred)
                    else:
                        oob_score_dict[index] = [Y_pred]
        
            sorted_oob_score_list = sorted(oob_score_dict.items(), key = lambda items:int(items[0]))
        
            for tup in sorted_oob_score_list:
                y_pred_oob.append(np.median(tup[1]))
                
            oob_score = mse(self.y,np.array(y_pred_oob))            
            print("The oob score for the random forest regressor on the training data is : {}".format(np.round(oob_score,4)))
        
    def rf_predict(self,query_pt):
        self.query_pt = query_pt
        if self.col_samp:
            list_of_predictions = []
            for i in range(len(self.mod_list)):
                y_pred = self.mod_list[i].predict(np.array(query_pt).reshape(1,-1)[:,self.list_selected_columns[i]])
                list_of_predictions.append(y_pred)
            print("The final prediction for the given query point is {}".format(np.median(list_of_predictions)))
        else:
            list_of_predictions = []
            for i in range(len(self.mod_list)):
                y_pred = self.mod_list[i].predict(np.array(query_pt).reshape(1,-1))
                list_of_predictions.append(y_pred)
            print("The final prediction for the given query point is {}".format(np.median(list_of_predictions)))
        
        
