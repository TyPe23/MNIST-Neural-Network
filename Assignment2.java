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

	public static int[][] training_data;
	public static int[][] testing_data;
	public static int numLayers;
	public static double[][] biases = new double[2][];
	public static double[][][] weights = new double[3][][];

	// start here
	public static void main(String[] args) throws Exception{

		// create a new file from the .csv training file
		Scanner train = new Scanner(new File(System.getProperty("user.dir") + "\\mnist_train.csv"));
		Scanner test = new Scanner(new File(System.getProperty("user.dir") + "\\mnist_test.csv"));

		// set the delimiter to a comma
		train.useDelimiter(",");
		test.useDelimiter(",");

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

		// close the .csv file before exiting
		train.close();
		test.close();
		int[] nodesPerLayer = {784,15,10};
		Network(nodesPerLayer);
	}

	// function that creates the weights and biases for the network
	public static void Network(int[] nodes) {
		numLayers = nodes.length;
		
		// for loop that creates the biases
		for(int i = 0; i < numLayers - 1; i++) {
			// allocates the memory for each layers' biases
			biases[i] = new double[nodes[i + 1]];

			// for loop that assigns values to each of the biases
			for(int j = 0; j < nodes[i + 1]; j++) {

				// random values from -1 to 1
				biases[i][j] = Math.random() * 2 - 1;
				System.out.print(biases[i][j] + " ");
			}
			System.out.println("");
		}

		// for loop that creates the weights
		for(int i = 0; i < numLayers - 1; i++) {
			// allocates the memory for each layers' weights
			weights[i] = new double[nodes[i]][nodes[i + 1]];
			
			for(int j = 0; j < nodes[i]; i++) {

			}
		}
	}
}