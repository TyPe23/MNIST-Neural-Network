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

		/* // print out all of the values to make sure it works
		int i = 0;

		while (i < 783 * 3 + 4) {
			if (train.hasNext()) {
				System.out.print(train.next());
			}
			i++;
		}

		// close the .csv file before exiting
		train.close();
		test.close(); */
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
			for(int j = 0; j < nodes[i + 1]; j++) {
				biases[i][j] = Math.random() * 2 - 1;
				System.out.print(biases[i][j] + " ");
			}
			System.out.println("");
		}

		// for loop that creates the weights
		for(int i = 0; i < numLayers - 1; i++) {
			// allocates the memory for each layers' weights
			weights[i] = new double[nodes[i]][nodes[i + 1]];
			System.out.println(weights.length);
			//for(int j = 0; j < )
		}
	}
}