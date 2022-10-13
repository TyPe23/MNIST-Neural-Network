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

// main class
public class Assignment2 {

	// 2D arrays containing training and testing data
	public static int[][] training_data = new int[60000][785];
	public static int[][] testing_data = new int[10000][785];

	// indices of training data array
	public static int[] training_indices = new int[60000];

	// multidementional arrays of weights and biases
	public static double[][] biases = new double[2][];
	public static double[][][] weights = new double[2][][];

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
			training_data[i][j] = Integer.valueOf(train.next());
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
			testing_data[i][j] = Integer.valueOf(test.next());
			j++;

			if (j == 785){
				i++;
				j = 0;
			}
		}

		i = 0;
		while (i < 60000) {
			training_indices[i] = i;
		}

		// close the .csv file before exiting
		train.close();
		test.close();
		int[] nodesPerLayer = {784,15,10};
		Network(nodesPerLayer);
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
			}
		}

		// weights
		// for loop that iterates through each layer
		for(int i = 0; i < nodes.length - 1; i++) {
			// allocates the memory for each layers' weights
			weights[i] = new double[nodes[i]][nodes[i + 1]];

			System.out.print("Layer " + i);

			// for loop that iterates through each node
			for(int j = 0; j < weights[i].length; j++) {
				
				System.out.print(" Node: " + j);
				// for loop that iterates through each weight
				for(int k = 0; k < weights[i][j].length; k++) {
					// random value from -1 to 1
					weights[i][j][k] = Math.random() * 2 - 1;
				}
				System.out.print(" Weights: " + weights[i][j].length);
			}
			System.out.println("");
		}
	}
}