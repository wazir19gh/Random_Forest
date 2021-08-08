# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 15:09:44 2021

@author: Wazir
"""

import numpy as np
from sklearn.metrics import accuracy_score as acc
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

class Random_Forest_Classifier:
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
        
        # Selecting the column indices anywhere between 3 and 13
        if self.col_samp:
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
                DT = DecisionTreeClassifier(max_depth = None)
                DT.fit(self.list_input_data[i],self.list_output_data[i])
                self.mod_list.append(DT)
            
            # Initializing a list to store the predictions of the n base learners.
            y_pred_list = []
            for i in range(len(self.mod_list)):
                Y_pred = self.mod_list[i].predict(self.x[:,self.list_selected_columns[i]]).reshape(-1,1)
                y_pred_list.append(Y_pred)
            
            # Converting the list of predictions to an array with each element of the array being a vector of predictions of n base learners
            y_pred_arr = np.array(y_pred_list)
            fin_y_pred = []
            
            # Getting the final predictions of each of the data point using majority voting
            for i in range(y_pred_arr.shape[1]):
                temp = pd.Series(y_pred_arr[:,i]).value_counts().reset_index()
                row_index = np.argmax(temp.iloc[:,1])
                predicted_class = temp.iloc[row_index,0]
                fin_y_pred.append(predicted_class)        
            
            # Getting the training accuracy
            train_accuracy = acc(self.y,np.array(fin_y_pred).reshape(-1,1))
            
            # Printing the training mean squared error
            print("The accuracy on the training data of the random forest classifier is : {}".format(np.round(train_accuracy,4)))
        
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
                DT = DecisionTreeClassifier(max_depth = None)
                DT.fit(self.list_input_data[i],self.list_output_data[i])
                self.mod_list.append(DT)
            
            # Initializing a list to store the predictions of the n base learners.
            y_pred_list = []
            for i in range(len(self.mod_list)):
                Y_pred = self.mod_list[i].predict(self.x).reshape(-1,1)
                y_pred_list.append(Y_pred)
            
            # Converting the list of predictions to an array with each element of the array being a vector of predictions of n base learners
            y_pred_arr = np.array(y_pred_list)
            fin_y_pred = []
            
            # Getting the final predictions of each of the data point using majority voting
            for i in range(y_pred_arr.shape[1]):
                temp = pd.Series(y_pred_arr[:,i]).value_counts().reset_index()
                row_index = np.argmax(temp.iloc[:,1])
                predicted_class = temp.iloc[row_index,0]
                fin_y_pred.append(predicted_class)
            
            # Getting the training mse
            train_accuracy = acc(self.y,np.array(fin_y_pred))
            
            # Printing the training mean squared error
            print("The accuracy on the training data of the random forest classifier is : {}".format(np.round(train_accuracy,4)))
            
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
        
            # Majority Vote
            for tup in sorted_oob_score_list:
                temp = pd.Series(tup[1]).value_counts().reset_index()
                row_index = np.argmax(temp.iloc[:,1])
                predicted_class = temp.iloc[row_index,0]
                y_pred_oob.append(predicted_class)
                
            oob_score = acc(self.y,np.array(y_pred_oob))            
            print("The oob score for the random forest classifier on the training data is : {}".format(np.round(oob_score,4)))
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
        
            # Majority Vote
            for tup in sorted_oob_score_list:
                temp = pd.Series(tup[1]).value_counts().reset_index()
                row_index = np.argmax(temp.iloc[:,1])
                predicted_class = temp.iloc[row_index,0]
                y_pred_oob.append(predicted_class)
                
            oob_score = acc(self.y,np.array(y_pred_oob))            
            print("The oob score for the random forest classifier on the training data is : {}".format(np.round(oob_score,4)))
            
    def rf_predict(self,query_pt):
        self.query_pt = query_pt
        if self.col_samp:
            list_of_predictions = []
            for i in range(len(self.mod_list)):
                y_pred = self.mod_list[i].predict(np.array(query_pt).reshape(1,-1)[:,self.list_selected_columns[i]])
                list_of_predictions.append(y_pred)
            temp = pd.Series(list_of_predictions).value_counts().reset_index()
            row_index = np.argmax(temp.iloc[:,1])
            predicted_class = temp.iloc[row_index,0]
            print("The final prediction for the given query point is {}".format(predicted_class))
        else:
            list_of_predictions = []
            for i in range(len(self.mod_list)):
                y_pred = self.mod_list[i].predict(np.array(query_pt).reshape(1,-1))
                list_of_predictions.append(y_pred)
            temp = pd.Series(list_of_predictions).value_counts().reset_index()
            row_index = np.argmax(temp.iloc[:,1])
            predicted_class = temp.iloc[row_index,0]
            print("The final prediction for the given query point is {}".format(predicted_class))
        
            
        
        