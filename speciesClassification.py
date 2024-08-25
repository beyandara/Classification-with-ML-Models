
import matplotlib.pyplot as plt
from matplotlib import lines
import numpy as np
import scipy.stats 
import binarytree
import csv
 
#########################################
#   Data input / prediction evaluation
#########################################

def read_data(csv_file):
    """ Read Palmer penguin data from CSV file and remove rows with missing
        data 
    # Returns:
    X:  Numpy array, shape (n_samples,4), where n_samples is number of rows 
        in the dataset. Contains the four numeric columns in the dataset 
        (bill length, bill depth, flipper length, body mass).
        Each column (each feature) is normalized by subtracting the column 
        mean and dividing by the column std.dev. ("z-score")
    y:  Numpy array, shape (n_samples,) 
        Contains integer value representing the penguin species, encoded as 
        follows (alphabetically according to species name):
            'Adelie':       0
            'Chinstrap':    1
            'Gentoo':       2
    """
    _, data = read_dataset_csv(csv_file)

    # appending element in first column of data, except if it includes 'NA'
    y = [i[0] for i in data if 'NA' not in i]
    # returns y where categories are numeric
    y, _ = encode_category_numeric(y) 

    # retrieving columns bill length, bill depth, flipper length and body mass
    X = [[float(value) for value in i[2:6]] for i in data if 'NA' not in i]
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) # normalizing data

    return X, y

def convert_y_to_binary(y,y_value_true):
    """ Helper function to convert integer valued y to binary (0 or 1) valued
        vector 
    
    # Input arguments:
    y:              Integer valued NumPy vector, shape (n_samples,)
    y_value_true    Value of y which will be converted to 1 in output.
                    All other values are converted to 0.

    # Returns:
    y_binary:   Binary vector, shape (n_samples,)
                1 for values in y that are equal to y_value_true, 0 otherwise
    """
    # converts y -> 1 if element matches with True index of y-value_true
    return np.where(y == y_value_true, 1, 0)


def train_test_split(X,y,train_frac):
    """ Split dataset into training and testing datasets

    # Input arguments:
    X:              Dataset, shape (n_samples,n_features)   
    y:              Values to be predicted, shape (n_samples)
    train_frac:     Fraction of data to be used for training
    
    # Returns:
    (X_train,y_train):  Training dataset 
    (X_test,y_test):    Test dataset
    """
    # randomizing X and y the same way
    p = np.random.permutation(len(X)) 
    X, y = X[p], y[p]

    # calculating the index for splitting training- and testing data
    train_split_index = round(len(X)*train_frac)
    test_split_index = len(X) - train_split_index 

    # creating a training dataset
    X_train = X[:train_split_index]
    y_train = y[:train_split_index]
    
    # creating a testing dataset
    X_test = X[-test_split_index:]
    y_test = y[-test_split_index:]

    # returning data as tuplet og Xs and ys
    train_data = (X_train, y_train)
    test_data = (X_test, y_test)

    return train_data, test_data

def accuracy(y_pred,y_true):
    """ Calculate accuracy of model based on predicted and true values 
    
    # Input arguments:
    y_pred:     Numpy array with predicted values, shape (n_samples,)
    y_true:     Numpy array with true values, shape (n_samples,)

    # Returns:
    accuracy:   Fraction of cases where the predicted values 
                are equal to the true values. Number in range [0,1]
    """
    # accuracy = true predictions / total predictions
    return np.sum(y_pred == y_true) / y_pred.shape[0]

###########################################################
#   Scatter_plot er hentet fra DecisionTree_intro.ipynb
###########################################################

def scatter_plot_2d_features(X,y,y_min=0,y_max=3,newfig=True):
    """ Visualize 2D features X and class label y as scatter plot
    
    # Arguments:
    X:    NumPy array with features. Shape (n_samples,2)
    y:    NumPy array with integer class labels. Shape (n_samples,)
    
    # Keyword arguments:
    y_min:   Minimum value for class label (usually zero) 
    y_max:   Maximum value for class label (usually n_classes-1)
    """
    if newfig:
        fig = plt.figure(figsize=(5,3.5))
    for label_value in np.unique(y):
        plt.scatter(x=X[y==label_value,0],
                    y=X[y==label_value,1],
                    c=y[y==label_value],
                    label=f'Class {label_value}',
                    vmin=y_min,
                    vmax=y_max)    
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.legend()

###################
#   Perceptron
###################

class Perceptron:
    """ Perceptron model for classifying two linearly separable classes """

    def __init__(self, num_attributes = 2):
        """ Initializing perceptron """ 
        self.bias = 0
        self.converged = False
        self.weights = np.random.rand(num_attributes)
        
    def predict(self, x):
        """ Predict / calculate perceptron output for single observation / row 
        
        # Input arguments:
        x:      Numpy array, shape (n_features,)
                Corresponds to single row of data matrix X   
        
        # Returns:
        f:      Activation function output based on x: 
                I = (weights * x - bias)
                f(I) = 1 if I > 0,  0 otherwise
        """
        # dot product of weights and input array - bias
        I = np.dot(self.weights, x) - self.bias
        
        # activation function returns 1 if I > 0, else 0
        return np.where(I > 0, 1, 0)
    

    def train(self, X, y, learning_rate=0.3, max_epochs=5):
        """ Fit perceptron to training data X with binary labels y
            
        # Input arguments:
        X:              2D NumPy array, shape (n_samples, n_features)
        y:              NumPy vector, shape (n_samples), with 0 and 1 
                        indicating which class each sample in X corresponds
                        to ("true" labels).
        learning_rate:  Learning rate, number in range (0.0 - 1.0)
        max_epochs:     Maximum number of epochs (integer, 1 or larger).
        tolerance:      Tolerance for total weight change to consider
                        convergence.
        """

        epoch = 0
        while epoch < max_epochs:
            # Assuming converged variable is True at the start of each epoch
            converged = True
           
            for x_ind, x_i in enumerate(X): # Iterating over every training row
                # predicting the class, v, of current data
                v = self.predict(x_i)
                if v != y[x_ind]:   # check if prediction is False           
                    converged = False 

                    # updating weights and bias
                    update = learning_rate * (y[x_ind] - v)
                    self.weights += update * x_i
                    self.bias += update * (-1)
            # if one epoch without fail; converged = True
            if converged:
                self.converged = True
                break 
            epoch += 1
        

    def print_info(self):
        """Helper function for printing perceptron weights and convergence"""
        print("Weights: " + str(self.weights))
        print("Bias: " + str(self.bias))
        print("Is converged: " + str(self.converged))
        

    def get_line_x_y(self):
        """ Helper function for calculating slope and intercept for decision 
            boundary 
        
        # Returns:
        slope:      Slope of decision boundary line, assuming first feature as
                    x value and second feature as y value.  
        intercept:  Value where decision boundary line crosses y axis. 
        """
        slope = -(self.weights[0]/self.weights[1])
        intercept = (self.bias / self.weights[1])

        return (slope,intercept)


##############################
#   Gini impurity functions  #
##############################

def gini_impurity(y):
    """ Calculate Gini impurity of a vector

    # Arguments:
    y   - 1D NumPy array with class labels

    # Returns:
    impurity  - Gini impurity, scalar in range [0,1)   
    """
    gini_impurity = 0

    # iterating through unique class labels in y
    for c in np.unique(y):
        # calculates probability for occurence of the current class
        prob = y[y == c].shape[0]/y.shape[0] 

        # adds gini impurity for the current class
        gini_impurity += prob*(1-prob)
    return gini_impurity

def gini_impurity_reduction(y,left_mask):
    """ Calculate the reduction in mean impurity from a binary split
        
    # Arguments:
    y           - 1D numpy array
    left_mask   - 1D numpy boolean array, True for "left " elements, False for
                  "right" 

    # Returns:
    impurity_reduction: Reduction in mean Gini impurity, scalar in range [0,0.5] 
                        Reduction is measured as _difference_ between Gini 
                        impurity for the original (not split) dataset, and the 
                        mean impurity for the two split datasets ("left" and 
                        "right").

    """
    # retrieving "left" and "right" datasett based om mask
    left = y[left_mask]
    right = y[~left_mask]

    # gini impurity for the dataset nd the splits
    y_gini = gini_impurity(y)
    left_gini = gini_impurity(left)
    right_gini = gini_impurity(right)

    # calculating mean gini impurity for splits
    weighted_gini = left_gini*(left.shape[0]/y.shape[0]) + \
                        right_gini*(right.shape[0]/y.shape[0])
    
    # returning reduction of impurity
    return y_gini-weighted_gini


def best_split_feature_value(X,y):
    """ Find feature and value "split" that yields highest impurity reduction
    
    # Arguments:
    X:       NumPy feature matrix, shape (n_samples, n_features)
    y:       NumPy class label vector, shape (n_samples,)
    
    # Returns:
    best_GI_reduction:     Reduction in Gini impurity for best split.
                            Zero if no split that reduces impurity exists. 
    feature_index:          Index of X column with best feature for split.
                            None if impurity_reduction = 0.
    feature_value:          Value of feature in X yielding best split of y
                            Dataset is split using 
                            X[:,feature_index] <= feature_value
                            None if impurity_reduction = 0.
    """
    
    best_GI_reduction =-np.inf
    feature_index = None
    feature_value = None

    # Iterate over each feature 
    for j in range(X.shape[1]):  # Assuming X is a 2D numpy array

        # Get unique values in the current feature
        unique_values = np.unique(X[:, j])

        # Iterate over each unique value in the feature
        for value in unique_values:

            # creating mask that says which element is <= value
            left_mask = X[:, j] <= value

            # calculating Gini impurity
            impurity_reduction = gini_impurity_reduction(y, left_mask)
            if impurity_reduction > best_GI_reduction:
                best_GI_reduction = impurity_reduction
                feature_index = j
                feature_value = value

    # return impurity_reduction, feature_index, feature_value
    return best_GI_reduction, feature_index, feature_value


###################
#   Node classes 
###################
class DecisionTreeBranchNode(binarytree.Node):
    def __init__(self, feature_index, feature_value, left=None, right=None):
        """ Initialize decision node
        
        # Arguments:
        feature_index    Index of X column used in question
        feature_value    Value of feature used in question
        left             Node, root of left subtree
        right            Node, root of right subtree
        """
        question_string = f'f{feature_index} <= {feature_value:.3g}'  # "General" format - fixed point/scientific
        super().__init__(value=question_string,left=left,right=right)
        self.feature_index = feature_index
        self.feature_value = feature_value


class DecisionTreeLeafNode(binarytree.Node):
    def __init__(self, y_value):
        """ Initialize leaf node

        # Arguments:
        y_value     class in dataset (e.g. integer or string) represented by leaf
        """ 
        super().__init__(str(y_value))
        self.y_value = y_value


####################
#   Decision tree  #
####################
class DecisionTree():
    def __init__(self):
        """ Initialize decision tree (no arguments)
        """
        self._root = None
        self._y_dtype = None


    def __str__(self):
        """ Return string representation of decision tree (based on binarytree.Node.__str__())"""
        if self._root is not None:
            return(str(self._root))
        else:
            return '<Empty decision tree>'


    def train(self,X,y):
        """ Train decision tree based on labelled dataset 
        
        # Arguments:
        X        NumPy feature matrix, shape (n_samples, n_features)
        y        NumPy class label vector, shape (n_samples,)
        
        """
        self._y_dtype = y.dtype
        self._root = self._build_tree(X,y)

    
    def _build_tree(self,X,y):
        """ Recursively build decision tree
        
        # Arguments:
        X        NumPy feature matrix, shape (n_samples, n_features)
        y        NumPy class label vector, shape (n_samples,)

        """
        # find best binary split of dataset
        impurity_reduction, feature_index, feature_value = best_split_feature_value(X,y) 

        # If impurity can't be reduced any more, create and return leaf node
        if impurity_reduction == 0:
            leaf_value = scipy.stats.mode(y,keepdims=False)[0] # Use most common class in dataset
            return DecisionTreeLeafNode(leaf_value)
        # If impurity can be reduced, split dataset, build left and right 
        # branches, and return branch node.
        else:
            left_mask = X[:,feature_index] <= feature_value
            left = self._build_tree(X[left_mask],y[left_mask])
            right = self._build_tree(X[~left_mask],y[~left_mask])
            return DecisionTreeBranchNode(feature_index,feature_value,left,right)
    

    def predict(self,X):
        """ Predict class (y vector) for feature matrix X
        
        # Arguments:
        X        NumPy feature matrix, shape (n_samples, n_features)
        
        # Returns:
        y        NumPy class label vector (predicted), shape (n_samples,)
        """
        return self._predict(X,self._root)
    

    def _predict(self,X,node):
        """ Predict class (y-vector) for feature matrix X
        
        # Arguments:
        X       NumPy feature matrix, shape (n_samples, n_features)
        node    Node used to process the data. If the node is a leaf node,
                the data is classified with the value of the leaf node.
                If the node is a branch node, the data is split into left
                and right subsets, and classified by recursively calling
                _predict() on the left and right subsets.
        
        # Returns:
        y        NumPy class label vector (predicted), shape (n_samples,)
        """
        if isinstance(node, DecisionTreeLeafNode):
            # existing leaf node
            return node.y_value
        else:
            # Continue with the existing logic for branching
            left_mask = X[:, node.feature_index] <= node.feature_value
            left_predictions = self._predict(X[left_mask], node.left)
            
            # Use left_mask again to place predictions in the correct positions
            # This replaces the use of ~left_mask when predicting for the right subset
            right_predictions = self._predict(X[~left_mask], node.right)
            all_predictions = np.zeros(X.shape[0])
            
            all_predictions[left_mask] = left_predictions
            all_predictions[~left_mask] = right_predictions
            return all_predictions


### Oppgave 5 og 6 ###
def perceptron_gentoo(X, y):
    # feature selection (X) and class selection (y)
    X = X[:,(1,2)]  # bill depth and flipper length
    y = convert_y_to_binary(y, 2) # Gentoo

    # dividing data to training and testing split
    train_data, test_data= train_test_split(X, y, 0.75)
    X_train, y_train = train_data
    X_test, y_test = test_data

    # Creating an instance of the Perceptron model for Gentoo penguins
    perceptron_gentoo = Perceptron()
    perceptron_gentoo.train(X_train, y_train,
                            learning_rate=0.001, max_epochs=10000)  
    
    # making prediction on test dataset
    y_gentoo_pred = np.array([perceptron_gentoo.predict(x) for x in X_test])
    # Calculating accuracy
    accuracy_gentoo = accuracy(y_gentoo_pred, y_test)

    # printing information about the trained model
    perceptron_gentoo.print_info()
    print("accuracy: ", accuracy_gentoo)

    # Correct indices to select Gentoo and non-Gentoo data points
    is_gentoo = X_test[y_gentoo_pred == 1]  
    not_gentoo = X_test[y_gentoo_pred == 0]  

    # Plotting the decision boundary
    plt.scatter(is_gentoo[:, 0], is_gentoo[:, 1], label='Gentoo Penguins')
    plt.scatter(not_gentoo[:, 0], not_gentoo[:, 1], label='Non-Gentoo Penguins')  

    # Plotting decision boundary
    line = perceptron_gentoo.get_line_x_y()
    x_plot = np.array([-4, 4])
    y_plot = x_plot * line[0] + line[1]  # y = mx + c
    plt.plot(x_plot, y_plot, label='Decision Boundary')
    plt.legend()

    # labels and grid
    plt.xlabel('Bill Depth (mm)')
    plt.ylabel('Flipper Length (mm)')
    plt.grid()

    plt.show()


### Oppgave 7 ###
def perceptron_chinstrap(X, y):
    # feature selection (X) and class selection (y)
    X = X[:,(0,3)]  # bill length and body mass
    y = convert_y_to_binary(y, 1) # Chinstrap

    # dividing data to training and testing split
    train_data, test_data= train_test_split(X, y, 0.75)
    X_train, y_train = train_data
    X_test, y_test = test_data

    # Creating an instance of the Perceptron model for chinstrap penguins
    perceptron_chinstrap = Perceptron()
    perceptron_chinstrap.train(X_train, y_train,
                            learning_rate=0.001, max_epochs=1000)  
    
    # making prediction on test dataset
    y_chinstrap_pred = np.array([perceptron_chinstrap.predict(x) for x in X_test])
    # Calculating accuracy
    accuracy_chinstrap = accuracy(y_chinstrap_pred, y_test)

    # printing information about the trained model
    perceptron_chinstrap.print_info()
    print("accuracy: ", accuracy_chinstrap)

    # Correct indices to select chinstrap and non-chinstrap data points
    is_chinstrap = X_test[y_chinstrap_pred == 1]  
    not_chinstrap = X_test[y_chinstrap_pred == 0]  

    # Plotting the decision boundary
    plt.scatter(is_chinstrap[:, 0], is_chinstrap[:, 1], label='chinstrap Penguins')
    plt.scatter(not_chinstrap[:, 0], not_chinstrap[:, 1], label='Non-chinstrap Penguins')  

    # Plotting decision boundary
    line = perceptron_chinstrap.get_line_x_y()
    x_plot = np.array([-4, 4])
    y_plot = x_plot * line[0] + line[1]  # y = mx + c
    plt.plot(x_plot, y_plot, label='Decision Boundary')
    plt.legend()

    # labels and grid
    plt.xlabel('Bill Depth (mm)')
    plt.ylabel('Flipper Length (mm)')
    plt.grid()

    plt.show()

def decisiontree_gentoo(X, y):
    # feature selection (X) and class selection (y)
    X = X[:,(1,2)]  
    y = convert_y_to_binary(y, 2)

    # dividing data to training and testing split
    train_data, test_data= train_test_split(X, y, 0.75)
    X_train, y_train = train_data
    X_test, y_test = test_data

    # Creating an instance of the DecisionTree model for Gentoo penguins
    dt_gentoo = DecisionTree()
    dt_gentoo.train(X_train, y_train)

    # making prediction on test dataset & Calculating accuracy   
    y_gentoo_pred = dt_gentoo.predict(X_test)
    accuracy_gentoo = accuracy(y_gentoo_pred, y_test)
    
    # Visualize the decision tree structure & information about the trained model 
    print(dt_gentoo)
    print("Accuracy:", accuracy_gentoo)

    # Plotting datapoints
    scatter_plot_2d_features(X_test, y_test)
    plt.title("Decision-Tree model for Gentoo")
    plt.show()

def decisiontree_chinstrap(X, y):
    # feature selection (X) and class selection (y)
    X = X[:,(0,3)]  
    y = convert_y_to_binary(y, 1)

    # dividing data to training and testing split
    train_data, test_data= train_test_split(X, y, 0.75)
    X_train, y_train = train_data
    X_test, y_test = test_data

    # Creating an instance of the DecisionTree model for Chinsstrap penguins
    dt_chinstrap = DecisionTree()
    dt_chinstrap.train(X_train, y_train)

    # Visualize the decision tree structure
    print(dt_chinstrap)

    # making prediction on test dataset & Calculating accuracy
    y_chinstrap_pred = dt_chinstrap.predict(X_test)
    accuracy_chinstrap = accuracy(y_chinstrap_pred, y_test)
    
    # Visualize the decision tree structure & information about the trained model 
    print(dt_chinstrap)    
    print("Accuracy:", accuracy_chinstrap)

    # Plotting the datapoints
    scatter_plot_2d_features(X_test, y_test)
    plt.title("Decision-Tree for chinstrap")
    plt.show()

def decisiontree_species(X, y):

    # dividing data to training and testing split
    train_data, test_data= train_test_split(X, y, 0.75)
    X_train, y_train = train_data
    X_test, y_test = test_data

    # Creating an instance of the DecisionTree model for the penguins
    dt_species = DecisionTree()
    dt_species.train(X_train, y_train)

    # making prediction on test dataset & Calculating accuracy
    y_species_pred = dt_species.predict(X_test)
    accuracy_species = accuracy(y_species_pred, y_test)

    # Visualize the decision tree structure
    print(dt_species)
    print("Accuracy:", accuracy_species)

    # Plotting the decision boundary
    scatter_plot_2d_features(X_test, y_test)
    plt.title("Decision-Tree for all three species")
    plt.show()



#####################################################
# helper-methods from mandatory previous assigments #
#####################################################

def read_dataset_csv(csv_file_path, delimiter=','):
    """ Read dataset from CSV file and return as numpy array 
    
    # Input arguments:
    csv_file_path:  Path to CSV file (str or pathlib.Path)
    delimiter:      Delimiter used in file. ',' (default), ';' and ' '
                    are common choices. 

    # Returns:
    header:         List of column headers (strings)
                    Length (n_features,)
    X:              Numpy matrix, shape (n_samples, n_features),
                    with string datatype (e.g '<U6' for strings max. 6 chars long)        
    """
    data = []
    with open(csv_file_path, newline='', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        for line in csvreader:
            data.append(line)
    header = data[0]
    X = np.array(data[1:], dtype='U6')

    return header, X


def encode_category_numeric(string_array,dtype=int):
    """ Encode category strings as numbers
    
    # Input arguments:
    string_array:   NumPy array of strings representing categories in dataset
                    (e.g. ['dog','cat','dog','giraffe'])
    dtype:          Data type of numeric output array. 
                    Examples: int (default) or float. 
    
    # Returns:
    num_array:      Array of integers representing categories in string_array.
                    Each unique category is encoded with a single integer.
                    Integers are assigned according to the alphabetical 
                    ordering of the (unique) categories, starting from 0.
                    For string_array = ['dog','cat','dog','giraffe'], 
                    num_array = [1,0,1,2]
    category_index: Dictionary with keys correspoding to category names 
                    (strings) and values corresponding to category numbers.
                    For example above, category_index =
                    {'cat':0,'dog':1,'giraffe':2} 
    """

    category_index = {}
    unique_array = sorted(set(string_array))
    for number, category in enumerate(unique_array):
        category_index[category] = number
    num_array = np.array([category_index[x] for x in string_array])
    
    return num_array, category_index

if __name__ == '__main__':
    """ To run the Perceptron model and get the decision trees, 
        uncomment the lines below: Click on the line, 
        then press `Ctrl + /` (or `Cmd + /` on Mac) to uncomment. """
    
    X, y = read_data("palmer_penguins.csv")
    
    perceptron_gentoo(X, y)
    # perceptron_chinstrap(X, y)

    # decisiontree_gentoo(X, y)
    # decisiontree_chinstrap(X, y)
    # decisiontree_species(X, y)