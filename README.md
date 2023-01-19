![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)

# GA_CNN
Genetic Algorithm which optimizes Convolutional Neural Networks' architecture for image classification tasks.

## Content
- **`cnn.py`**: Contains the Convolutional Neural Network implementation. The architecture of each instance is defined depending on the parameters used at initialization time of the object.
- **`ga.py`**: In this file, the Genetic Algorithm is created. Each individual created by the algorithm contains a CNN architecture.
- **`main.py`**:  The dataset is defined (MNIST in this case, it can be changed but be aware to modify number of classes in ``cnn.py`` depending on the labels provided), an instance of the Genetic Algorithm is created and the algorithm is executed. Finally, the results of the algorithm are plotted in a graph.

## Usage
 - Just execute the ``main.py`` file or the *``main()``* function inside it.
 
Take into account that high computational resources are needed for obtaining good results  so, configure the parameters of the Genetic Algorithm depending on the resources available and your expectations with the execution.
 As recommendation, if looking for a fast execution just for experimentation, do not reduce the mutation parameter too much for the algorithm not to converge in early generations.
 Consider that if increasing excessively the maximum number of layers and the maximum kernel size,  it is possible that some of the resulting CNNs try to reduce the image below 1x1 and the algorithm will not be executed. Try to find a balance between the size of the images in the dataset and those parameters.