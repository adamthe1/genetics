#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random

import pandas as pd

with open("figure2.csv", 'r') as file:
    data = file.read().split('\n')[1:]

rows = [a.split(',') for a in data if a[0] == 'r']
heads = [data[i+1].split(',')[0] for i, a in enumerate(data) if a[0] == 'r'] 

for i in range(len(rows)):
    rows[i] = [b for b in rows[i] if b]
    rows[i].extend(rows[i].pop(-1).split(' '))
    rows[i] = [b for b in rows[i] if b]
    rows[i][0] = heads[i]
t_data = list(zip(*rows))

df = pd.DataFrame(t_data[1:], columns=t_data[0])
df[0:4]


# In[3]:


blue = [0]*2234
green = [1]*428
brown = [2]*324
data_dict = {}
data_dict['eye_color'] = blue + green + brown
col_count = [2234, 428, 324]

for pheno in df.columns:
    data_dict[pheno] = []
    for i, stat in enumerate(list(df[pheno][1:4])):
        amount = round(col_count[i]*(float(stat)/100))
        l = [1]*amount + [0]*(col_count[i]-amount)
        random.shuffle(l)
        data_dict[pheno].extend(l)
final_df = pd.DataFrame(data_dict)  
col_mv = final_df.pop("eye_color")
final_df.insert(9, "eye_color", col_mv)
eye_color = final_df['eye_color'].copy()
eye_color.loc[eye_color == 0] = "blue"
eye_color.loc[eye_color == 1] = "green"
eye_color.loc[eye_color == 2] = "brown"
eye_color



# In[ ]:


class Node:
    def __init__(self, 
                 feature_index=None,
                 threshold=None,
                 left_node=None,
                 right_node=None,
                 info_gain=None,
                 id=None,
                 depth=None,
                 value=None,
                 types=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.right_node = right_node
        self.left_node = left_node
        self.info_gain = info_gain
        self.id = id
        self.depth = depth
        self.types = types
        # for leaf nnode
        self.value = value


# In[154]:


class DecisionTree:
    def __init__(self, min_sample_split=30, max_depth=12):
        self.root = None

        #stopping conditions
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
    def build_tree(self, dataset, curr_depth=0, id=0):

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        #print(num_samples, num_features)

        if num_samples >= self.min_sample_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            #print(best_split["curr_gain"])
            #print(curr_depth)
            if best_split["curr_gain"] > 0:
            # left set
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1, id*2 + 1)
                #print(left_subtree)
                # right set
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1, id*2 + 2)
                

                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["curr_gain"]
                            ,depth=curr_depth, id=id, types=Y)

        leaf_value = self.get_leaf_value(Y)
        #print(leaf_value)
        return Node(value=leaf_value, depth=curr_depth, id=id, types=Y)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {"curr_gain": 0}
        max_infogain = -1000000000

        for feature_index in range(num_features):
            feature_values = dataset[:,feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold) 
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, y_left, y_right = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    info_gain = self.information_gain(y, y_left, y_right, "gini")
                    #print(info_gain)
                    if info_gain > max_infogain:
                        #print('yay')
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["curr_gain"] = info_gain
                        max_infogain = info_gain
        #rint(max_infogain)
        return best_split

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([sample for sample in dataset if sample[feature_index] <= threshold] )
        dataset_right = np.array([sample for sample in dataset if sample[feature_index] > threshold])
        return dataset_left, dataset_right        
    
    def information_gain(self, y, y_left, y_right, type):
        # weights might use later 
        weight_l = len(y_left)/len(y)
        weight_r = len(y_right)/len(y)
        if type == "gini":
            info_gain = self.gini_index(y) - (weight_l*self.gini_index(y_left) + weight_r*self.gini_index(y_right))
            #print(info_gain)
        # entropy maybe
        return info_gain

    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y==cls])/len(y)
            gini += p_cls**2
        #print(1- gini)
        return 1 - gini

    def get_leaf_value(self, y):
        values = list(y)
        return max(values, key=values.count)

    def fit(self, x, y):
        dataset = np.concatenate((x, y), axis=1)
        #print(dataset)
        self.root = self.build_tree(dataset)

    def print_tree(self, tree=None, depth=0, leaf_l=[], inner_l=[]):
        if not tree:
            tree = self.root
            print(self.root.feature_index)
        if tree.value is not None:
            leaf_l.append(tree)
            print("leaf: id="+str(tree.id) + " depth=" + str(tree.depth)+ " value="+ str(tree.value))
            
        else:
            inner_l.append(tree)
            print(" id=" + str(tree.id) + " depth=" + str(tree.depth) + " X_"+str(tree.feature_index), " thresh: "+str(tree.threshold), " info ", tree.info_gain)
            print(f"left:")
            leaf_l, inner_l = self.print_tree(tree.left_node, depth + 1, leaf_l, inner_l)
            print(f"right:")
            leaf_l, inner_l = self.print_tree(tree.right_node, depth + 1, leaf_l, inner_l)
        return leaf_l, inner_l
 
    # predict 1
    def make_prediction(self, x, tree):
        if tree.value != None:
            #print(tree.value)
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left_node)
        else:
            return self.make_prediction(x, tree.right_node)
            
    def predict(self, set, root=None):
        if not root:
            root = self.root
        predictions = [self.make_prediction(x, root) for x in set]
        return predictions
        # predict 1

    found = False
    def pruning(self, prunedList, tree=None):
        if self.found:
            return tree
        if tree == None:
            tree = self.root
        if tree.value:
            return tree
        #print(tree.id, prunedList)
        if int(tree.id) == prunedList[0]:
            print('hello', end=" ")
            tree.value = self.get_leaf_value(tree.types)
            self.found = True
            return tree
        self.pruning(prunedList, tree.left_node)
        if self.found:
            return tree

        self.pruning(prunedList, tree.right_node)
        if self.found:
            return tree

        return tree
        
    
  
        
        

X = final_df.iloc[:, :-1].values
Y = final_df.iloc[:, -1].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

classifier = DecisionTree()
classifier.fit(X_train, Y_train)

#classifier.print_tree()


# In[155]:


#classifier.print_tree()

Y_pred = list(classifier.predict(X_test))

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))

#print(len([1 for i, a in enumerate(Y_test) if a == 2 and Y_pred[i] == 2])/len([a for a in Y_test if a == 2]))

for type in [0, 1, 2]:
    print(len([1 for i, a in enumerate(Y_test) if a == type and Y_pred[i] == type])/len([a for a in Y_test if a == type]) )


# In[156]:


leaf, inner = classifier.print_tree()


# In[157]:


print([a.id for a in leaf])
l = [a.id for a in leaf]
print([a.id for a in inner
      ])


# 
# 
# pruning

# In[160]:


max_accuracy = accuracy_score(Y_test, Y_pred)
for node in inner:
    print(node.id, end=", ")
    if node.id in [6, 13]:
        continue
    temp = DecisionTree()
    temp.fit(X_train, Y_train)
    temp.pruning([node.id])
    temp_preds = list(temp.predict(X_test))
    accuracy = accuracy_score(Y_test, temp_preds)
    print(accuracy)
    if accuracy > max_accuracy:
        best_one = temp
        max_accuracy = accuracy
    


# In[ ]:


new


# In[ ]:




