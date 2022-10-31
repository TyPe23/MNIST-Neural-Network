/* 
	Ty Pederson
	10349042
	10/10/2022
	Neural Network Program

	Three layer, back propagated, sigmoid neural network
	Neural network is designed to solve the MNIST problem of correctly identifying handwritten digits
	Gives the user the ability to train their own network, 
	save that network to a file, 
	load a network from a file,
	and test their network on training and/or testing data
*/

import java.io.*;
import java.util.Scanner;
import java.lang.Math;
import java.util.Arrays;
import java.util.Random;


public class NeuralNetworkProgram {

	public static double[][] training_data = new double[60000][785];
	public static double[][] testing_data = new double[10000][785];

	
	public static int[] training_indices = new int[60000];

	
	public static int[] digits = {0,0,0,0,0,0,0,0,0,0};

	
	public static int[] digitsAns = {0,0,0,0,0,0,0,0,0,0};

	
	public static double[][] biases = new double[2][];
	public static double[][][] weights = new double[2][][];

	public static double[][] startingBiases = new double[2][];
	public static double[][][] startingWeights = new double[2][][];

	
	public static double[][] biasGradients = new double[2][];
	public static double[][][] weightGradients = new double[2][][];

	public static int[] nodesPerLayer = {784,100,10};
	public static int epochs = 30;
	public static float learningRate = 0.45f;
	public static float miniBatchSize = 10.0f;

	public static boolean seeASCII = false;
	public static boolean incorrectOnly = false;

	public static Scanner inputScanner = new Scanner(System.in);


	public static void main(String[] args) throws Exception{

		boolean networkLoaded = false;

		Scanner train = new Scanner(new File(System.getProperty("user.dir") + "\\mnist_train.csv"));
		Scanner test = new Scanner(new File(System.getProperty("user.dir") + "\\mnist_test.csv"));

		
		train.useDelimiter(",|\\n");
		test.useDelimiter(",|\\n");


		// put all of the values of the training and testing data into 2D arrays
		int val = 0;
		int line = 0;
		while (train.hasNext()) {

			training_indices[val] = val;

			training_data[val][line] = Double.valueOf(train.next()) / 255;

			if (test.hasNext()) {
				testing_data[val][line] = Double.valueOf(test.next()) / 255;
			}

			line++;

			if (line == 785){
				val++;
				line = 0;
			}
		}

		train.close();
		test.close();

		createNetwork(nodesPerLayer);	

		int userInput;
		int accuracy;


		do {
			System.out.println();
			System.out.println("[1] Train the network.");
			System.out.println("[2] Load a pre-trained network.");

			if (networkLoaded) {
				System.out.println("[3] Display network accuracy on training data.");
				System.out.println("[4] Display network accuracy on testing data.");
				System.out.println("[5] Save the network state to file.");
				System.out.println("[6] Toggle ASCII display of outputs when selecting options 3 or 4.");
				if (seeASCII) {
					System.out.println("[7] Toggle ASCII for incorrect values only.");
				}
			}

			System.out.println("[0] Exit.");
			System.out.println();

			// check if the input is an integer
			while (!inputScanner.hasNextInt()) {
				System.out.println("Please enter an integer.");
				System.out.println();
				inputScanner.next();
				System.out.println();
			}
			userInput = inputScanner.nextInt();
			System.out.println();


			// different actions based on user input
			switch (userInput) {
				// exits
				case 0:
					break;

				// trains
				case 1:
					networkLoaded = true;

					initialWB();

					for (int e = 0; e < epochs; e++) {

						SGD();

						accuracy = 0;

						for (int d = 0; d < digits.length; d++) {
							System.out.print(d + " = " + digitsAns[d] + "/" + digits[d] + " ");
							accuracy += digitsAns[d];
						}
						System.out.println("Accurracy = " + accuracy + "/60000 = " + calcPercent() + "%.");

						if (e < epochs){
							resetDigits();
						}
					}
					break;

				// loads
				case 2: 
					networkLoaded = true;
					SaveLoad.loadNetwork();
					break;
				
				// tests on training data
				case 3:
					if (networkLoaded) {
						feedForward(training_data);

						accuracy = 0;

						for (int d = 0; d < digits.length; d++) {
							System.out.print(d + " = " + digitsAns[d] + "/" + digits[d] + " ");
							accuracy += digitsAns[d];
						}
						System.out.println("Accurracy = " + accuracy + "/60000 = " + calcPercent() + "%.");

						resetDigits();
					}
					break;

				// tests on testing data
				case 4:
					if (networkLoaded) {
						feedForward(testing_data);

						accuracy = 0;

						for (int d = 0; d < digits.length; d++) {
							System.out.print(d + " = " + digitsAns[d] + "/" + digits[d] + " ");
							accuracy += digitsAns[d];
						}
						System.out.println("Accurracy = " + accuracy + "/10000 = " + calcPercent() + "%.");

						resetDigits();
					}
					break;

				// saves
				case 5:
					// checks if network has weights and biases
					if (networkLoaded) {
						SaveLoad.saveNetwork();
					}
					break;

				// toggles ASCII display
				case 6:
					seeASCII = !seeASCII;
					if (seeASCII) {
						System.out.println("ASCII vaules will be displayed.");
					}
					else {
						System.out.println("ASCII values will no longer be displayed.");
					}
					break;

				// toggles ASCII values shown
				case 7:
					if (seeASCII) {
						incorrectOnly = !incorrectOnly;
						if (incorrectOnly) {
							System.out.println("ASCII vaules will be displayed for incorrect values only.");
						}
						else {
							System.out.println("ASCII values will be displayed for all values.");
						}
					}
					break;

				// invalid input
				default:
					System.out.println("Please enter a valid input.");
					System.out.println();
					break;
			}
		} while (userInput != 0);

		inputScanner.close();
	}


	
	public interface SaveLoad {
		// loads saved weights and biases
		public static void loadNetwork() {
			try {
				ObjectInputStream loadWeights = new ObjectInputStream(new FileInputStream("savedWeights"));
				ObjectInputStream loadBiases = new ObjectInputStream(new FileInputStream("savedBiases"));

				weights = (double[][][])loadWeights.readObject();

				biases = (double[][])loadBiases.readObject();

				loadWeights.close();
				loadBiases.close();
			} 
			catch (IOException e) {
				System.out.println("File load error.");
			}
			catch (ClassNotFoundException c) {
				System.out.println("File read error.");
			}
		}


		// saves weights and biases to new files
		public static void saveNetwork() {
			try {
				ObjectOutputStream savedWeights = new ObjectOutputStream(new FileOutputStream("savedWeights"));
				ObjectOutputStream savedBiases = new ObjectOutputStream(new FileOutputStream("savedBiases"));

				savedWeights.writeObject(weights);
				savedBiases.writeObject(biases);

				savedWeights.close();
				savedBiases.close();

			} 
			catch (IOException e) {
				System.out.println("File creation error.");
			}
		}
	}


	// function that creates the weights and biases for the network
	public static void createNetwork(int[] nodes) {

		for(int lyr = 0; lyr < nodes.length - 1; lyr++) {

			biases[lyr] = new double[nodes[lyr + 1]];
			startingBiases[lyr] = new double[nodes[lyr + 1]];
			biasGradients[lyr] = new double[nodes[lyr + 1]];

			weights[lyr] = new double[nodes[lyr + 1]][nodes[lyr]];
			startingWeights[lyr] = new double[nodes[lyr + 1]][nodes[lyr]];
			weightGradients[lyr] = new double[nodes[lyr + 1]][nodes[lyr]];


			for(int nd = 0; nd < nodes[lyr + 1]; nd++) {

				// random values from -1 to 1 
				biases[lyr][nd] = Math.random() * 2 - 1;
				startingBiases[lyr][nd] = biases[lyr][nd];
				biasGradients[lyr][nd] = 0;

				for(int wt = 0; wt < weights[lyr][nd].length; wt++) {

					// random values from -1 to 1 
					weights[lyr][nd][wt] = Math.random() * 2 - 1;
					startingWeights[lyr][nd][wt] = weights[lyr][nd][wt];
					weightGradients[lyr][nd][wt] = 0;
				}
			}
		}
	}


	// performs Stochastic Gradient Descent
	public static void SGD() {

		int[] indices_copy = training_indices;
		Random rand = new Random();

		for(int i = 0; i < indices_copy.length; i++) {

			int randIndx = rand.nextInt(indices_copy.length);
			int temp = indices_copy[randIndx];
			indices_copy[randIndx] = indices_copy[i];
			indices_copy[i] = temp;
		}


		for (int i = 0; i < indices_copy.length;) {

			reset();

			for (int m = 0; m < miniBatchSize; m++, i++) {

				backProp(Arrays.copyOfRange(training_data[indices_copy[i]], 1, training_data[indices_copy[i]].length), indices_copy[i]);
			}

			updateWB();
		}
	}


	// performs back propagation
	public static void backProp(double[] a, int index) {
		
		double[][] activations = new double[biases.length + 1][]; 

		double[] OHV = oneHotVector(training_data[index][0]);

		activations[0] = a;

		for (int lyr = 0; lyr < biases.length; lyr++) {

			activations[lyr + 1] = new double[biases[lyr].length]; 

			for (int nd = 0; nd < weights[lyr].length; nd++) {

				double sumWA = 0;

				for (int wt = 0; wt < weights[lyr][nd].length; wt++) {
					
					sumWA += a[wt] * weights[lyr][nd][wt];
				}
				activations[lyr + 1][nd] = sigmoid(sumWA + biases[lyr][nd]);
			}
			a = activations[lyr + 1];
		}


		if (findMaxIndex(a) == training_data[index][0] * 255) {
			digitsAns[(int)(training_data[index][0] * 255)]++;
		}

		digits[(int)(training_data[index][0] * 255)]++;


		// output layer
		for(int nd = 0; nd < weights[1].length; nd++) {

				biasGradients[1][nd] += (activations[2][nd] - OHV[nd]) * activations[2][nd] * (1 - activations[2][nd]);

			for(int wt = 0; wt < weights[1][nd].length; wt++) {
				weightGradients[1][nd][wt] += activations[1][wt] * biasGradients[1][nd];
			}
		}


		// hidden layer
		for(int nd = 0; nd < weights[0].length; nd++) {

			double sumWB = 0;

			for(int wt = 0; wt < weights[1].length; wt++) {
				sumWB += weights[1][wt][nd] * biasGradients[1][wt];
			}

			biasGradients[0][nd] += sumWB * activations[1][nd] * (1 - activations[1][nd]);

			for(int wt = 0; wt < weights[0][nd].length; wt++) {
				weightGradients[0][nd][wt] += activations[0][wt] * biasGradients[0][nd];
			}
		}
	}


	// performs gradient updates
	public static void updateWB() {

		for(int lyr = 0; lyr < biases.length; lyr++) {

			for(int nd = 0; nd < biases[lyr].length; nd++) {

				biases[lyr][nd] -= learningRate/miniBatchSize * biasGradients[lyr][nd];

				for(int wt = 0; wt < weights[lyr][nd].length; wt++) {

					weights[lyr][nd][wt] -= learningRate/miniBatchSize * weightGradients[lyr][nd][wt];
				}
			}
		}
	}


	// performs feed forward pass on given testing data (not used in backProp function)
	public static void feedForward(double[][] dataSet) {

		for (int img = 0; img < dataSet.length; img++) {
			
			double[] input = Arrays.copyOfRange(dataSet[img], 1, dataSet[img].length);

			for (int lyr = 0; lyr < biases.length; lyr++) {

				double[] temp = new double[biases[lyr].length]; 

				for (int nd = 0; nd < weights[lyr].length; nd++) {

					double sumWA = 0;

					for (int wt = 0; wt < weights[lyr][nd].length; wt++) {

						sumWA += input[wt] * weights[lyr][nd][wt];
					}
					temp[nd] = sigmoid(sumWA + biases[lyr][nd]);
				}
				input = temp;
			}

			// update the number of correct answers
			if (findMaxIndex(input) == dataSet[img][0] * 255) {
				digitsAns[(int)(dataSet[img][0] * 255)]++;
				if (!incorrectOnly && seeASCII) {
					displayDigit(dataSet[img], img, findMaxIndex(input));
				}
			}
			else if (seeASCII) {
				displayDigit(dataSet[img], img, findMaxIndex(input));
			}
			digits[(int)(dataSet[img][0] * 255)]++;
		}
	}


	// creates one hot vector of expected
	public static double[] oneHotVector(double expectedOut) {

		double[] OHV = new double[biases[biases.length - 1].length];

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


	// finds the index of the larges value in the array
	public static int findMaxIndex(double[] ans) {

		double max = ans[0];
		int maxIndex = 0;

		for (int i = 1; i < ans.length; i++) {
			if (ans[i] > max) {
				max = ans[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}


	// resets weight and bias gradients to 0
	public static void reset() {

		for(int lyr = 0; lyr < biases.length; lyr++) {

			for(int nd = 0; nd < biases[lyr].length; nd++) {

				biasGradients[lyr][nd] = 0;

				for(int wt = 0; wt < weights[lyr][nd].length; wt++) {

					weightGradients[lyr][nd][wt] = 0;
				}
			}
		}
	}


	// resets weight and bias gradients initial values
	public static void initialWB() {

		for(int lyr = 0; lyr < biases.length; lyr++) {

			for(int nd = 0; nd < biases[lyr].length; nd++) {

				biases[lyr][nd] = startingBiases[lyr][nd];

				for(int wt = 0; wt < weights[lyr][nd].length; wt++) {

					weights[lyr][nd][wt] = startingWeights[lyr][nd][wt];
				}
			}
		}
	}


	// resets the values of correct numbers and total numbers per digit
	public static void resetDigits() {
		for (int i = 0; i < digitsAns.length; i++) {
			digitsAns[i] = 0;
			digits[i] = 0;
		}
	}


	// calculates the accuracy percentage
	public static double calcPercent() {
		float ans = 0.0f;
		float tot = 0.0f;

		for (int i = 0; i < digits.length; i++) {
			ans += digitsAns[i];
			tot += digits[i];
		}

		// two decimal places
		return Math.round(((ans / tot) * 100) * 100) / 100.0;
	}


	// sigmoid activation function
	public static double sigmoid(double z) {
		return 1 / (1 + Math.exp(-z));
	}


	// displays the digits if selected by the user
	public static void displayDigit(double[] input, int index, int output) {
		// array of possible ASCII outputs
		String[] symbol = {" ", ".", ",", ":", ";", "i", "Y", "H", "#", "&", "@"};

		System.out.print("Testing case #" + index + ": Correct classification = " + (int)(input[0] * 255) + " Network Output = " + output);

		// checks if the network got the right answer
		if ((input[0] * 255 == output)){
			System.out.print(" Correct.");
		}
		else {
			System.out.print(" Incorrect.");
		}
		System.out.println();

		// iterates through each pixel value
		for (int pv = 1; pv < input.length;) {

			// iterates through each line
			for (int l = 0; l < 28; l++, pv++) {
				System.out.print(symbol[(int) Math.floor(input[pv] * 10.0)]);
			}
			System.out.print("\n");
		}
		
		int userInput;

		System.out.println("[1] Continue.");
		System.out.println("[Any int] Finish without ASCII.");
		System.out.println();
		while (!inputScanner.hasNextInt()) {
			System.out.println("Please enter an integer.");
			System.out.println();
			inputScanner.next();
			System.out.println();
		}
		userInput = inputScanner.nextInt();

		if (userInput != 1) {
			seeASCII = false;
		}
		System.out.println();
	}
}