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

	// array of the number of each digit
	public static int[] digits = {0,0,0,0,0,0,0,0,0,0};

	// array of the correct answers of each digit
	public static int[] digitsAns = {0,0,0,0,0,0,0,0,0,0};

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

		// put all of the values of the training and testing data into 2D arrays
		int i = 0;
		int j = 0;
		while (train.hasNext()) {

			training_indices[i] = i;

			training_data[i][j] = Double.valueOf(train.next()) / 255;

			if (test.hasNext()) {
				testing_data[i][j] = Double.valueOf(test.next()) / 255;
			}

			j++;

			if (j == 785){
				i++;
				j = 0;
			}
		}

		/* for (int k = 0; k < 60000; k++) {
			digits[(int)(training_data[k][0] * 255)] += 1;
		} */

		// close the .csv file before exiting
		train.close();
		test.close();

		// create the network
		int[] nodesPerLayer = {784,15,10};
		Network(nodesPerLayer);	

		for (i = 0; i < 30; i++) {
			SGD();
			System.out.println(Arrays.toString(digitsAns) + " " + calcPercent() + "%");
			if (i < 29){
				resetDigits();
			}
		}
		System.out.println(Arrays.toString(digits));
	}

	// function that creates the weights and biases for the network
	public static void Network(int[] nodes) {

		// for loop that iterates through each layer
		for(int i = 0; i < nodes.length - 1; i++) {

			// allocates the memory for bias
			biases[i] = new double[nodes[i + 1]];
			biasGradients[i] = new double[nodes[i + 1]];

			// allocates the memory for weight
			weights[i] = new double[nodes[i + 1]][nodes[i]];
			weightGradients[i] = new double[nodes[i + 1]][nodes[i]];

			// iterates through each input for the node
			for(int j = 0; j < nodes[i + 1]; j++) {

				// random values from -1 to 1 
				biases[i][j] = Math.random() * 2 - 1;
				biasGradients[i][j] = 0;

				// for loop that iterates through each weight
				for(int k = 0; k < weights[i][j].length; k++) {

					// random value from -1 to 1
					weights[i][j][k] = Math.random() * 2 - 1;
					weightGradients[i][j][k] = 0;
				}
			}
		}
	}

	// performs Stochastic Gradient Descent
	public static void SGD() {

		// make a copy of the training indices as to not scramble the original
		int[] indices_copy = training_indices;
		Random rand = new Random();

		// randomize training indicies
		for(int i = 0; i < indices_copy.length; i++) {

			int randIndx = rand.nextInt(indices_copy.length);
			int temp = indices_copy[randIndx];
			indices_copy[randIndx] = indices_copy[i];
			indices_copy[i] = temp;
		}

		int miniBatchSize = 10;
		// iterates through training data
		for (int j = 0; j < indices_copy.length;) {

			// sets gradients to 0
			reset();

			// iterates through mini batch
			for (int k = 0; k < miniBatchSize; k++, j++) {

				// back propagation of the current piece of training data
				backProp(Arrays.copyOfRange(training_data[indices_copy[j]], 1, training_data[indices_copy[j]].length), indices_copy[j]);
			}

			// updates weights and biases
			updateWB();

			//System.out.println();
		}
	}

	// performs back propagation
	public static void backProp(double[] a, int index) {
		
		// array of activations for each layer
		double[][] activations = new double[biases.length + 1][]; 

		// one hot vector of expected output
		double[] OHV = oneHotVector(training_data[index][0]);

		// set first layer to input
		activations[0] = a;

		// iterates through layers
		for (int i = 0; i < biases.length; i++) {

			// initializes activations array of current layer
			activations[i + 1] = new double[biases[i].length]; 

			// iterates through nodes
			for (int j = 0; j < weights[i].length; j++) {

				// starting value of sum of activation values times weights
				double sumWA = 0;

				// iterates through each input for the node
				for (int k = 0; k < weights[i][j].length; k++) {
					
					// add to sum of activation values times weights
					sumWA += a[k] * weights[i][j][k];
				}
				// set the result of the sigmoid to the current index of activations
				activations[i + 1][j] = sigmoid(sumWA + biases[i][j]);
			}
			// replace the input array with the activation values of the current layer
			a = activations[i + 1];
		}

		// update the number of correct answers
		if (findMaxIndex(a) == training_data[index][0] * 255) {
			digitsAns[(int)(training_data[index][0] * 255)]++;
		}

		digits[(int)(training_data[index][0] * 255)]++;

		// Lth layer
		// iterates through nodes
		for(int j = 0; j < weights[1].length; j++) {

				// add to bias gradient
				biasGradients[1][j] += (activations[2][j] - OHV[j]) * activations[2][j] * (1 - activations[2][j]);

			// add to weight gradients
			for(int k = 0; k < weights[1][j].length; k++) {
				weightGradients[1][j][k] += activations[1][k] * biasGradients[1][j];
			}
		}


		// lth layers
		// iterates through nodes
		for(int j = 0; j < weights[0].length; j++) {

			// starting value of sum of weights times bias gradient
			double sumWB = 0;

			// add to sum of weights times bias gradient
			for(int k = 0; k < weights[1].length; k++) {
				sumWB += weights[1][k][j] * biasGradients[1][k];
			}

			biasGradients[0][j] += sumWB * activations[1][j] * (1 - activations[1][j]);


			// add to weight gradients
			for(int k = 0; k < weights[0][j].length; k++) {
				weightGradients[0][j][k] += activations[0][k] * biasGradients[0][j];
			}
		}
	}

	// performs gradient updates
	public static void updateWB() {

		// iterates through layers
		for(int i = 0; i < biases.length; i++) {

			// iterates through nodes
			for(int j = 0; j < biases[i].length; j++) {

				// update bias
				biases[i][j] -= 3.0/10.0 * biasGradients[i][j];

				// iterates through weights
				for(int k = 0; k < weights[i][j].length; k++) {

					// update weight
					weights[i][j][k] -= 3.0/10.0 * weightGradients[i][j][k];
				}
			}
		}
	}

	// performs feed forward pass on given testing data (not used in backProp function)
	public static double[] feedForward(double[] a) {

		// iterates through layers
		for (int i = 0; i < biases.length; i++) {

			// temporary array of doubles with a length equal to the number of nodes in the layer
			double[] temp = new double[biases[i].length]; 

			// iterates through node
			for (int j = 0; j < weights[i].length; j++) {

				// starting value of sum of activation values times weights
				double sumWA = 0;

				// iterates through input for the node
				for (int k = 0; k < weights[i][j].length; k++) {

					// add to sum of activation values times weights
					sumWA += a[k] * weights[i][j][k];
				}
				// set the result of the sigmoid to the current index of temp
				temp[j] = sigmoid(sumWA + biases[i][j]);
			}
			// replace the input array with the activation values of the current layer
			a = temp;
		}
		// returns the activation values of the final layer of the network
		return a;
	}

	// creates one hot vector of expected
	public static double[] oneHotVector(double expectedOut) {

		// initialize empty one hot vector
		double[] OHV = new double[biases[biases.length - 1].length];

		// iterates through OHV indices
		for (int i = 0; i < OHV.length; i++) {

			// set to 1 if index matches output
			if (i == expectedOut * 255) {
				OHV[i] = 1;
			}
			// set to 0 otherwise
			else {
				OHV[i] = 0;
			}
		}
		return OHV;
	}

	// finds the index of the larges value in the array
	public static int findMaxIndex(double[] ans) {

		// set starting values for max and max index
		double max = ans[0];
		int maxIndex = 0;

		// iterate through ans
		for (int i = 1; i < ans.length; i++) {

			// update max and max index if new max is found
			if (ans[i] > max) {
				max = ans[i];
				maxIndex = i;
			}
		}

		return maxIndex;
	}

	// resets weight and bias gradients to 0
	public static void reset() {

		// iterates through layers
		for(int i = 0; i < biases.length; i++) {

			// iterates through nodes
			for(int j = 0; j < biases[i].length; j++) {

				//System.out.println(i + " " + biasGradients[i][j]);
				biasGradients[i][j] = 0;

				//System.out.println();
				// iterates through each input for the node
				for(int k = 0; k < weights[i][j].length; k++) {

					//System.out.println(weightGradients[i][j][k]);
					weightGradients[i][j][k] = 0;
				}
			}
			//System.out.println();
		}
	}

	public static void resetDigits() {
		for (int i = 0; i < digitsAns.length; i++) {
			digitsAns[i] = 0;
			digits[i] = 0;
		}
	}

	public static float calcPercent() {
		float ans = 0.0f;
		float tot = 0.0f;

		for (int i = 0; i < digits.length; i++) {
			ans += digitsAns[i];
			tot += digits[i];
		}

		return (ans / tot) * 100;
	}

	// sigmoid activation function
	public static double sigmoid(double z) {
		return 1 / (1 + Math.exp(-z));
	}
}