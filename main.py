from ga import GeneticAlgorithm
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms

def main():

    # Download train and test data and transform to pytorch tensor
    batch_size = 64
    train_data = datasets.MNIST(
        root='input/data',
        train=True,
        download=True,
        transform= transforms.ToTensor())
    test_data = datasets.MNIST(
        root='input/data',
        train=False,
        download=True,
        transform=transforms.ToTensor())
    
    # Reduce dataset for shorter training time
    train_data = Subset(train_data, indices=range(len(train_data) // 10))
    # Reduce dataset for shorter testing time
    #test_data = Subset(test_data, indices=range(len(train_data) // 10))

    train_load = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True)
    test_load = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True)

    # Create an instance of the Genetic Algorithm
    ga = GeneticAlgorithm(10, 20, 10, 3, 3, 3, 64, 2, 0.9, 0.4, 1, train_load, test_load)
    # Execute the algorithm
    ga.exec()

    # Plot the results
    import matplotlib.pyplot as plt
    import seaborn as sns
    maxfitnessValues = ga.sols_max_fits
    meanFitnessValues = ga.sols_avg_fits
    sns.set_style("whitegrid")
    plt.plot(maxfitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.plot(xlabel='Generation', ylabel='Max / Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.show()

if __name__ == "__main__":
    main()