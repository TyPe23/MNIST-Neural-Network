/* 
	Ty Pederson
	10349042
	10/10/2022
	Assignment 2

	DON'T FORGET THE DESCRIPTION
*/


// import necessary packages
import java.io.*;
import java.util.Scanner;
import java.lang.Math;
import java.util.Arrays;
import java.util.Random;

// main class
public class Assignment2 {

	// 2D arrays containing training and testing data
	public static double[][] training_data = new double[60000][785];
	public static double[][] testing_data = new double[10000][785];

	// indices of training data array
	public static int[] training_indices = new int[60000];

	// multidementional arrays of weights and biases
	public static double[][] biases = new double[2][];
	public static double[][][] weights = new double[2][][];

	// multidementional arrays of weight and bias gradients
	public static double[][] biasGradients = new double[2][];
	public static double[][][] weightGradients = new double[2][][];

	// start here
	public static void main(String[] args) throws Exception{

		// create a new file from the .csv training file
		Scanner train = new Scanner(new File(System.getProperty("user.dir") + "\\mnist_train.csv"));
		Scanner test = new Scanner(new File(System.getProperty("user.dir") + "\\mnist_test.csv"));

		// set the delimiter to a comma
		train.useDelimiter(",|\\n");
		test.useDelimiter(",|\\n");

		// put all of the values of the training data into a 2D array
		int i = 0;
		int j = 0;
		while (train.hasNext()) {
			training_data[i][j] = Double.valueOf(train.next()) / 255;
			j++;

			if (j == 785){
				i++;
				j = 0;
			}
		}

		// put all of the values of the testing data into a 2D array
		i = 0;
		j = 0;
		while (test.hasNext()) {
			testing_data[i][j] = Double.valueOf(test.next()) / 255;
			j++;

			if (j == 785){
				i++;
				j = 0;
			}
		}

		i = 0;
		while (i < 60000) {
			training_indices[i] = i;
			i++;
		}

		// close the .csv file before exiting
		train.close();
		test.close();

		// create the network
		int[] nodesPerLayer = {784,15,10};
		Network(nodesPerLayer);

		SGD();
	}

	// function that creates the weights and biases for the network
	public static void Network(int[] nodes) {
		// biases
		// for loop that iterates through each layer
		for(int i = 0; i < nodes.length - 1; i++) {
			// allocates the memory for each layers' biases
			biases[i] = new double[nodes[i + 1]];

			// for loop iterates through each node
			for(int j = 0; j < nodes[i + 1]; j++) {
				// random values from -1 to 1 
				biases[i][j] = Math.random() * 2 - 1;
				biasGradients[i][j] = 0;
			}
		}

		// weights
		// for loop that iterates through each layer
		for(int i = 0; i < nodes.length - 1; i++) {
			// allocates the memory for each layers' weights
			weights[i] = new double[nodes[i + 1]][nodes[i]];

			//System.out.println("Layer " + i);

			// for loop that iterates through each node
			for(int j = 0; j < weights[i].length; j++) {
				
				//System.out.print(" N: " + j);
				// for loop that iterates through each weight
				for(int k = 0; k < weights[i][j].length; k++) {
					// random value from -1 to 1
					weights[i][j][k] = Math.random() * 2 - 1;
					weightGradients[i][j][k] = 0;
				}
				//System.out.print(" W: " + weights[i][j].length + "\n");
			}
			//System.out.println();
		}
	}

	public static void reset() {
		for(int i = 0; i < biases.length; i++) {
			for(int j = 0; j < biases[i].length; j++) {
				biasGradients[i][j] = 0;
			}
		}

		for(int i = 0; i < biases.length - 1; i++) {
			for(int j = 0; j < weights[i].length; j++) {
				for(int k = 0; k < weights[i][j].length; k++) {
					weightGradients[i][j][k] = 0;
				}
			}
		}
	}

	// function that performs Stochastic Gradient Descent to train the network
	public static void SGD() {
		// randomize the training set by moving around the indicies
		// make a copy of the training set as to not scramble the original
		int[] training_indices_copy = training_indices;
		Random rand = new Random();

		for(int i = 0; i < training_indices_copy.length; i++) {
			int randIndx = rand.nextInt(training_indices_copy.length);
			int temp = training_indices_copy[randIndx];
			training_indices_copy[randIndx] = training_indices_copy[i];
			training_indices_copy[i] = temp;
		}

		int miniBatchSize = 10;
		for (int j = 0; j < training_indices_copy.length;) {

			reset();

			for (int k = 0; k < miniBatchSize; k++, j++) {
				backProp(Arrays.copyOfRange(training_data[training_indices_copy[j]], 1, training_data[training_indices_copy[j]].length), training_indices_copy[j]);

				//System.out.println(Arrays.toString(outputs[k]));
			}
		}
	}

	// THIS IS WHAT YOU NEED TO WORK ON NOW
	public static void backProp(double[] a, int index) {
		
		double[][] temp = new double[biases.length][]; 

		// for loop that goes through each layers
		for (int i = 0; i < biases.length; i++) {

			temp[i] = new double[biases[i].length]; 

			// for loop that goes through each node
			for (int j = 0; j < weights[i].length; j++) {

				double sumWA = 0;

				// for loop that goes through each input for the node
				for (int k = 0; k < weights[i][j].length; k++) {

					sumWA += a[k] * weights[i][j][k];
				}
				temp[i][j] = sigmoid(sumWA + biases[i][j]);
			}
			a = temp[i];
		}
		
		

		for(int i = biases.length - 1; i > 0; i--) {
			for(int j = 0; j < weights[i].length; j++) {

				double sumWB = 0;

				for(int k = 0; k < weights[i][j].length; k++) {
					sumWB += weightGradients[i + 1][j][k] * biasGradients[i + 1][j];
				}

				if (i != biases.length - 1) {
					biasGradients[i][j] = sumWB * temp[i][j] * (1 - temp[i][j]);
				}
				else {
					biasGradients[i][j] = (temp[i][j] - oneHotVector(training_data[index][0])[j]) * temp[i][j] * (1 - temp[i][j]);
				}
			}
		}
	}

	// function that returns a one hot vector from the input of the expected output for the handwritten number
	public static int[] oneHotVector(double expectedOut) {
		int[] OHV = new int[biases[biases.length - 1].length];

		for (int i = 0; i < OHV.length; i++) {
			if (i == expectedOut * 255) {
				OHV[i] = 1;
			}
			else {
				OHV[i] = 0;
			}
		}

		return OHV;
	}

	// feed forward function that takes an input of an array of 784 doubles which represents one of the pixels of a hand written number
	public static double[] feedForward(double[] a) {
		// for loop that goes through each layers
		for (int i = 0; i < biases.length; i++) {
			// temporary array of doubles with a length equal to the number of nodes in the layer
			double[] temp = new double[biases[i].length]; 
			// for loop that goes through each node
			for (int j = 0; j < weights[i].length; j++) {
				// starting value of 0 for the summation of activation values times their weights
				double sumWA = 0;
				// for loop that goes through each input for the node
				for (int k = 0; k < weights[i][j].length; k++) {
					// multiply the activation value times the weight and add it to the sum
					sumWA += a[k] * weights[i][j][k];
				}
				// set the result of the sigmoid to the current index of temp
				temp[j] = sigmoid(sumWA + biases[i][j]);
			}
			// replace the input array which was the activation values of the previous layer 
			// with the temp array which represents the activation values of the current layer
			// so that the values can either be returned or used for the next layer
			a = temp;
		}
		// returns the activation values of the final layer of the network
		return a;
	}

	// sigmoid activation function
	public static double sigmoid(double z) {
		return 1 / (1 + Math.exp(-z));
	}
}