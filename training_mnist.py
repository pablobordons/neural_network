import numpy as np
import struct
import os


class ANN(object):
    ''' 
    Artificial Network Class. 
    '''
    
    def __init__(self, X, y, bias=True, n_neurons=5,  
                 learning_rate=1, Lambda=0, n_batches=1, 
                 verbose=True, random_seed=10):
        '''
        Parameters:
            X: Data to train the model 
               Instances as rows and features as columns
            y: Desired output of the network
            
            bias: True to use bias in the input and in the hidden layer
            n_neurons: number of neurons in the hidden layer
            learning_rate: parameter that multiplies the gradient matrices
                           to control the learning steps
            Lambda: regularization parameter to smooth out spiky values 
                    (prevents overfitting but decreases accuracy)
            n_batches: number of batches X is divided by when training.
                       1 batch to use Batch gradient descent
                       n batches to use mini-batch gradient descent
                       n=n_instances to use stochastic gradient descent (online)
            verbose: True to output feedback when training
            random_seed: to control the reproducibility.
        '''
        
        if n_batches > X.shape[0]:
            raise ValueError("The number of batches cannot be larger than the number of inputs")

        # initialize basic parameters of the ANN:
        self.bias = bias
        self.X = X
        self.y = y
        if bias:
            self.X = self._add_bias(self.X)        # add an extra column if needed
        
        self.n_inputs = self.X.shape[0]            # number of inputs (rows)
        self.n_features = self.X.shape[1]          # number of features (columns)
        self.n_outputs = self.y.shape[1]           # number of rows in the output
        self.n_neurons = n_neurons                 # number of neurons in the hidden layer
        
        self.learning_rate = learning_rate         # controls the learning rate 
        self.Lambda = Lambda                       # controls the regularization
        
        self.n_batches = n_batches                 # controls the gradient descent mode
        self.batch_size = int(self.n_inputs/self.n_batches)   # n of inputs in each batch
        self._init_batches()                       # initializes the batches (list of dict)

        self.random_seed = random_seed             # controls the reproducibility 
        self.verbose = verbose                     # controls the feedback durng training
        self.train_step = 0 
        
        self.costs = [0]                           # store the cost function at each step 
        self._init_weights()                       # initialize the weights of the neurons 

     
        
    # ---------     Main Logic Methods   --------------- #

    def _forward_backard(self):
        '''
        This method contains the main algorithm for both:
            feeding the network (forward)  
            training the network (backward)
        
        It computes the values for every batch and stores them at
        self.batches, which is a list of dictionaries.
        
        Finally, it computes the cost associated to each batch.
        '''
        for batch in self.batches:
            
            # Forward 
            batch["z2"] = np.dot(batch["X"], self.W1)
            batch["a2"] = self.sigmoid(batch["z2"])
            if self.bias:
                batch["a2"] = self._add_bias(batch["a2"])
        
            batch["z3"] = np.dot(batch["a2"], self.W2)
            batch["a3"] = self.sigmoid(batch["z3"])
            
            # Backward
            batch["e3"] = batch["y"] - batch["a3"]
            batch["d3"] = batch["e3"]*self.sigmoid_prime(batch["a3"])
            
            batch["e2"] = np.dot(batch["d3"], self.W2.T)
            batch["d2"] = batch["e2"]*self.sigmoid_prime(batch["a2"])
            
            batch["D2"] = np.dot(batch["a2"].T, batch["d3"])
            batch["D1"] = np.dot(batch["X"].T, batch["d2"]) 
            if self.bias:
                batch["D1"] = batch["D1"][:,1:] # remove the extra bias column
                
            # Cost
            batch["cost"] = (sum((batch["a3"] - batch["y"])**2)*0.5/self.n_inputs)[0]
    
    
    def _update_weights(self):
        '''
        Method to update the weights W1 and W2 using the Deltas D1 and D2
        stored in each batch (computing the mean of all of them).
        It also takes care of the regularization (if lambda != 0)
        '''
        regularization_1 = self.Lambda*self.W1
        regularization_2 = self.Lambda*self.W2
        
        # get the average of D1 and D2 store at each batch
        D1 = sum([batch["D1"] for batch in self.batches])/len(self.batches)
        D2 = sum([batch["D2"] for batch in self.batches])/len(self.batches)
        
        # update the weights
        self.W1 += self.learning_rate*(D1 - regularization_1)
        self.W2 += self.learning_rate*(D2 - regularization_2)
    
    
    def _update_cost(self):
        '''
        Method to compute the cost 
        as an average of all the costs stored in each batch
        '''
        cost = sum([b["cost"] for b in self.batches])/self.n_batches
        self.costs.append(cost)
       
    
    def _update_output(self):
        '''
        Method to store the output in the output layer a3 in a single value 
        to ease its acces from outside the class
        '''
        self.a3 =  np.concatenate([batch["a3"] for batch in self.batches])
    
    
    # --------- Functional Methods ----------- #
    
    def train_once(self):
        '''Wrap of the training methods for one iteration'''
        self._forward_backard()   # main algorithm
        self._update_weights()    # gradient descent
        self._update_cost()       # cost for each epoch
        self._update_output()     # save output layer values
        self.train_step += 1     
        
    def train(self, train_steps=10000):
        '''Wrap to carry one training routine, controling the epochs (steps)'''
        for _ in range(train_steps):
            self.train_once()
            if self.verbose and not self.train_step%int(train_steps/5):
                print("iteration: {}, cost: {:.3E}".format(self.train_step, self.costs[-1]))
  

    def predict(self, X):
        '''
        Method to facilitate the feeding of the network with data not contained in the model
        in rder to compute a prediction value that is returned.
        It basically computes a forward routine without storing any value,
        using the weights of the network W1 and W2
        '''
        if self.bias:
            X = self._add_bias(X)
        
        if X.shape[1] != self.n_features:
            raise ValueError("The feature dimensions do not match with \
                              the ones used to train the model")
        
        z2 = np.dot(X, self.W1)
        a2 = self.sigmoid(z2)
        if self.bias:
            a2 = self._add_bias(a2)
    
        z3 = np.dot(a2, self.W2)
        a3 = self.sigmoid(z3)
        
        return a3
        
    
    # --------    Initialization methods   ----------------- #
    
    def _init_batches(self):
        '''
        Method to initialize the batches as a list of dictionaries.
        Each batches contains its index 
        '''
        self.batches = [{
            "n": batch_number,
            "X": self._batch_matrix(self.X, self.batch_size, batch_number),
            "y": self._batch_matrix(self.y, self.batch_size, batch_number),
            "z2": None, "a2": None, "z3": None, "a3": None, # init fordward parameters
            "e3": None, "d3": None, "e2": None, "d2": None, # init backprop parameters
            "D1": None, "D2": None                          # init gradient parameters
        } for batch_number in range(self.n_batches)]
    
    def _init_weights(self):
        '''initilize the weights W1 and W2'''
        np.random.seed(self.random_seed)
        
        # weight matrix W1 to go from the input layer to the hidden layer
        self.W1 = np.random.random((self.n_features, self.n_neurons))
        
        # weight matrix W2 to go from the hidden layer to the output layer
        b = 1 if self.bias else 0  # add an extra bias column
        self.W2 = np.random.random((self.n_neurons + b, self.n_outputs))
   


    # --------    Helper methods   ------------------------ #
    
    def sigmoid(self, t):
        '''computes the sigmoid function'''
        return 1/(1+np.exp(-t))

    def sigmoid_prime(self, t):
        '''computes the derivative of the sigmoid function'''
        return self.sigmoid(t)*(1 - self.sigmoid(t))
    
    def _add_bias(self, A, axis=1):
        '''append a bias column (ones column) to a given matrix A'''
        bias = np.ones(len(A)).reshape(len(A), 1)
        return np.append(bias, A, axis=axis)
    
    def _batch_matrix(self, A, batch_size, batch_number, axis=0):
        '''returns a chopped matrix along one axis.
        NOTE: automatically returns the leftover batch 
        when passed the last batch number'''
        offset = batch_number*batch_size
        n = offset + batch_size
        
        # chop through rows or chop through columns
        if axis == 0: # chop rows
            return A[offset:n,:]
        if axis == 1: # chop columns
            return A[:,offset:n,]


def load_mnist(data_path=None):

    if data_path is None:
        data_path = os.path.dirname(os.path.realpath(__file__))

    # ------   TRAIN SET    ------- #

    # labels
    labels_train_path = os.path.join(data_path, "train-labels-idx1-ubyte")
    with open(labels_train_path, "rb") as labels_file:
        struct.unpack(">II", labels_file.read(8))
        labels_train = np.fromfile(labels_file, dtype=np.int8)

    # Images
    images_train_path = os.path.join(data_path, "train-images-idx3-ubyte")
    with open(images_train_path, 'rb') as imgs_file:
        _, _, rows, cols = struct.unpack(">IIII", imgs_file.read(16))
        images_train = np.fromfile(imgs_file, dtype=np.uint8).reshape(len(labels_train), rows, cols)


    # ------   TEST SET    ------- #

    # labels
    labels_test_path = os.path.join(data_path, "t10k-labels-idx1-ubyte")
    with open(labels_test_path, "rb") as labels_file:
        struct.unpack(">II", labels_file.read(8))
        labels_test = np.fromfile(labels_file, dtype=np.int8)

    # Images
    images_test_path = os.path.join(data_path, "t10k-images-idx3-ubyte")
    with open(images_test_path, 'rb') as imgs_file:
        _, _, rows, cols = struct.unpack(">IIII", imgs_file.read(16))
        images_test = np.fromfile(imgs_file, dtype=np.uint8).reshape(len(labels_test), rows, cols)

    return {"train": {"images": images_train, "labels": labels_train}, 
            "test": {"images": images_test, "labels": labels_test}}


def expand_label(label):
    new_label = np.zeros(10)
    new_label[label] = 1
    return new_label

def get_max(row):
    return max(range(len(row)), key=row.__getitem__)


def train_routine(n_train=1000, offset_train=0, train_steps=1000,
                  mnist_dataset=None,
                  return_model=True, return_accuracy=True,
                  ann_parameters={"n_neurons": 50, "n_batches": 100}):

    if type(mnist_dataset) != dict:
        raise ValueError("Please, pass the MNIST dataset as a dictionary.")


    # Load the dataset into local variables
    images_train = mnist_dataset["train"]["images"]
    images_test = mnist_dataset["test"]["images"]

    labels_train = mnist_dataset["train"]["labels"]
    labels_test = mnist_dataset["test"]["labels"]

    print("Preproccessing the dataset...", end=" ")
    # ----- Training set ----- #
    train_X = images_train[offset_train:offset_train+n_train]
    train_X = np.array([i.flatten() for i in train_X])
    train_X = np.array([img / max(img) for img in train_X]) # normalize
    
    ann_parameters["X"] = train_X
    
    train_y = labels_train[offset_train:offset_train+n_train]
    train_y = np.array([train_y]).T
    train_y = np.array([expand_label(y) for y in train_y])
    
    ann_parameters["y"] = train_y
    
    # ----- Testing set ----- #
    test_X = images_test
    test_X = np.array([i.flatten() for i in test_X])
    test_X = np.array([img / max(img) for img in test_X]) # normalize

    test_y = labels_test
    test_y = np.array([test_y]).T
    test_y = np.array([expand_label(y) for y in test_y])
    print("OK")

    # ----- Train the network ------ #
    print("Initializing the Neural Network...", end=" ")
    ann = ANN(**ann_parameters)

    print("OK\n\nTraining the Neural Network ({} Epochs)...".format(train_steps))

    ann.train(train_steps=train_steps)
    print("Network training Finished.\nComputing the accuracy...")

    print("\n# --------- Accuracy ----------- #\n")
    # Accuracy in the training
    y_predicted_train = ann.a3
    y_predicted_train = [get_max(y) for y in y_predicted_train]

    acc_train = sum(y_predicted_train == labels_train[offset_train:offset_train+n_train])/n_train*100
    print("Training Accuracy: {}%".format(round(acc_train,2)))

    # Accuracy in the test
    y_predicted_test = ann.predict(test_X)
    y_predicted_test = [get_max(y) for y in y_predicted_test]

    acc_test = sum(y_predicted_test == labels_test)/test_X.shape[0]*100
    print("Test Accuracy: {}%".format(round(acc_test)))

    output = {}
    
    if return_model:
        output["model"] = ann
        output["model_param"] = ann_parameters
    
    if return_accuracy:
        output["accuracy"] = {"train": acc_train, "test": acc_test}
        
    if output:
        return output


if __name__ == "__main__":

    print("\n\n\tArtificial Neural Network training over the MNIST digit dataset.\n\n")

    path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(path, "data")
    
    print("Loading MNIST...", end=" ")
    dataset = load_mnist(data_path)
    print("OK")

    ann_parameters = {
        "bias": True,
        "n_neurons": 50, 
        "n_batches": 2000,
        "learning_rate": 0.9, 
        "Lambda": 0.002,
        "random_seed": 10
    }

    control_output = train_routine(n_train=20000, train_steps=200, ann_parameters=ann_parameters, 
                                   mnist_dataset=dataset)

    print("\nThe parameters used to train the network are:\n")
    print("\tBias: {}".format(ann_parameters["bias"]))
    print("\tNeurons in hidden layer: {}".format(ann_parameters["n_neurons"]))
    print("\tNumber of batches: {}".format(ann_parameters["n_batches"]))
    print("\tLearning Rate: {}".format(ann_parameters["learning_rate"]))
    print("\tLambda: {}".format(ann_parameters["Lambda"]))
    print("\tThe random weights were initialized using the seed {}".format(ann_parameters["random_seed"]))
    print("\n#{}#".format("-"*70))
    print("\n\n\t\tEND\n\n")
