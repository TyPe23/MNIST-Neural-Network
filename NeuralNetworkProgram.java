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


// import necessary packages
import java.io.*;
import java.util.Scanner;
import java.lang.Math;
import java.util.Arrays;
import java.util.Random;

// main class
public class NeuralNetworkProgram {

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

	public static double[][] startingBiases = new double[2][];
	public static double[][][] startingWeights = new double[2][][];

	// multidementional arrays of weight and bias gradients
	public static double[][] biasGradients = new double[2][];
	public static double[][][] weightGradients = new double[2][][];

	public static int epochs = 30;

	public static boolean seeASCII = false;
	public static boolean incorrectOnly = false;

	public static Scanner inputScanner = new Scanner(System.in);

	// start here
	public static void main(String[] args) throws Exception{

		// boolean value reflecting if the network has weights and biases
		boolean networkLoaded = false;

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

		// close the .csv file before exiting
		train.close();
		test.close();

		// create the network
		int[] nodesPerLayer = {784,100,10};
		createNetwork(nodesPerLayer);	

		int userInput;
		int accuracy;

		// loop that asks for user input
		do {
			// displays options to the user
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
					// changes bool to true allowing values 3, 4, & 5
					networkLoaded = true;

					// resets the weights and biases to their starting values to retrain the network
					resetWB();

					// iterates 
					for (i = 0; i < epochs; i++) {

						// performs stochastic gradient descent
						SGD();

						// resets the accuracy to 0
						accuracy = 0;

						// prints out the number correct/total number of each digit and updates accuracy
						for (int k = 0; k < digits.length; k++) {
							System.out.print(k + " = " + digitsAns[k] + "/" + digits[k] + " ");
							accuracy += digitsAns[k];
						}
						// prints accuracy
						System.out.println("Accurracy = " + accuracy + "/60000 = " + calcPercent() + "%.");

						// resets the digit totals and correct answers
						if (i < epochs){
							resetDigits();
						}
					}
					break;

				// loads
				case 2: 
					// changes bool to true allowing values 3, 4, & 5
					networkLoaded = true;
					loadNetwork();
					break;
				
				// tests on training data
				case 3:
					// checks if network has weights and biases
					if (networkLoaded) {
						// performs feed forward passes over training data
						feedForward(training_data);

						// resets accuracy to 0
						accuracy = 0;

						// prints out the number correct/total number of each digit and updates accuracy
						for (int k = 0; k < digits.length; k++) {
							System.out.print(k + " = " + digitsAns[k] + "/" + digits[k] + " ");
							accuracy += digitsAns[k];
						}
						// prints accuracy
						System.out.println("Accurracy = " + accuracy + "/60000 = " + calcPercent() + "%.");

						// resets the digit totals and correct answers
						resetDigits();
					}
					
					break;

				// tests on testing data
				case 4:
					// checks if network has weights and biases
					if (networkLoaded) {
						// performs feed forward passes over training data
						feedForward(testing_data);

						// resets accuracy to 0
						accuracy = 0;

						// prints out the number correct/total number of each digit and updates accuracy
						for (int k = 0; k < digits.length; k++) {
							System.out.print(k + " = " + digitsAns[k] + "/" + digits[k] + " ");
							accuracy += digitsAns[k];
						}
						// prints accuracy
						System.out.println("Accurracy = " + accuracy + "/10000 = " + calcPercent() + "%.");

						// resets the digit totals and correct answers
						resetDigits();
					}
						
					break;

				// saves
				case 5:
					// checks if network has weights and biases
					if (networkLoaded) {
						saveNetwork();
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
			// exits if input is 0
		} while (userInput != 0);

		// closes input scanner before exiting
		inputScanner.close();
	}


	// loads saved weights and biases
	public static void loadNetwork() {
		try {
			// opens input streams of weights and biases files
			ObjectInputStream loadWeights = new ObjectInputStream(new FileInputStream("savedWeights"));
			ObjectInputStream loadBiases = new ObjectInputStream(new FileInputStream("savedBiases"));

			// loads saved weights from input stream to weights 
			weights = (double[][][])loadWeights.readObject();

			// loads saved biases from input stream to biases
			biases = (double[][])loadBiases.readObject();

			// closes input streams
			loadWeights.close();
			loadBiases.close();
		} 
		// error handling for file load and file read errors
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
			// opens output streams for weights and biases files
			ObjectOutputStream savedWeights = new ObjectOutputStream(new FileOutputStream("savedWeights"));
			ObjectOutputStream savedBiases = new ObjectOutputStream(new FileOutputStream("savedBiases"));

			// writes weights and biases to files
			savedWeights.writeObject(weights);
			savedBiases.writeObject(biases);

			// closes output streams
			savedWeights.close();
			savedBiases.close();

		} 
		// error handling for file creation error
		catch (IOException e) {
			System.out.println("File creation error.");
		}
	}


	// function that creates the weights and biases for the network
	public static void createNetwork(int[] nodes) {

		// for loop that iterates through each layer
		for(int i = 0; i < nodes.length - 1; i++) {

			// allocates the memory for bias, starting bias and bias gradient
			biases[i] = new double[nodes[i + 1]];
			startingBiases[i] = new double[nodes[i + 1]];
			biasGradients[i] = new double[nodes[i + 1]];

			// allocates the memory for weight, starting weight and weight gradient
			weights[i] = new double[nodes[i + 1]][nodes[i]];
			startingWeights[i] = new double[nodes[i + 1]][nodes[i]];
			weightGradients[i] = new double[nodes[i + 1]][nodes[i]];

			// iterates through each input for the node
			for(int j = 0; j < nodes[i + 1]; j++) {

				// random values from -1 to 1 
				biases[i][j] = Math.random() * 2 - 1;
				// sets starting bias to bias
				startingBiases[i][j] = biases[i][j];
				// sets bias gradient to 0
				biasGradients[i][j] = 0;

				// for loop that iterates through each weight
				for(int k = 0; k < weights[i][j].length; k++) {

					// random value from -1 to 1
					weights[i][j][k] = Math.random() * 2 - 1;
					// sets starting weight to weight
					startingWeights[i][j][k] = weights[i][j][k];
					// sets weight gradient to 0
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
			// creates random index
			int randIndx = rand.nextInt(indices_copy.length);
			// saves value at random index
			int temp = indices_copy[randIndx];
			// replaces value at random index
			indices_copy[randIndx] = indices_copy[i];
			// replaces value at index i with saved value
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

			// add to bias gradients
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
				biases[i][j] -= 0.45/10.0 * biasGradients[i][j];

				// iterates through weights
				for(int k = 0; k < weights[i][j].length; k++) {

					// update weight
					weights[i][j][k] -= 0.45/10.0 * weightGradients[i][j][k];
				}
			}
		}
	}


	// performs feed forward pass on given testing data (not used in backProp function)
	public static void feedForward(double[][] a) {

		for (int x = 0; x < a.length; x++) {
			
			double[] input = Arrays.copyOfRange(a[x], 1, a[x].length);

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
						sumWA += input[k] * weights[i][j][k];
					}
					// set the result of the sigmoid to the current index of temp
					temp[j] = sigmoid(sumWA + biases[i][j]);
				}
				// replace the input array with the activation values of the current layer
				input = temp;
			}
			// update the number of correct answers
			if (findMaxIndex(input) == a[x][0] * 255) {
				digitsAns[(int)(a[x][0] * 255)]++;
				if (!incorrectOnly && seeASCII) {
					displayDigit(a[x], x, findMaxIndex(input));
				}
			}
			// calls dislplay function if the user selected to display incorrect digits
			else if (seeASCII) {
				displayDigit(a[x], x, findMaxIndex(input));
			}
			// update the number of digits tried
			digits[(int)(a[x][0] * 255)]++;
		}
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

				biasGradients[i][j] = 0;

				// iterates through each input for the node
				for(int k = 0; k < weights[i][j].length; k++) {

					weightGradients[i][j][k] = 0;
				}
			}
		}
	}


	// resets weight and bias gradients to 0
	public static void resetWB() {

		// iterates through layers
		for(int i = 0; i < biases.length; i++) {

			// iterates through nodes
			for(int j = 0; j < biases[i].length; j++) {

				//System.out.println(i + " " + biasGradients[i][j]);
				biases[i][j] = startingBiases[i][j];

				//System.out.println();
				// iterates through each input for the node
				for(int k = 0; k < weights[i][j].length; k++) {

					//System.out.println(weightGradients[i][j][k]);
					weights[i][j][k] = startingWeights[i][j][k];
				}
			}
			//System.out.println();
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

		// returns percentage rounded to two decimal places
		return Math.round(((ans / tot) * 100) * 100) / 100.0;
	}

	// sigmoid activation function
	public static double sigmoid(double z) {
		return 1 / (1 + Math.exp(-z));
	}

	// displays the digits if selected by the user
	public static void displayDigit(double[] input, int index, int output) {
		// array of possible ASCII outputs
		String[] symbol = {" ", ".", ",", ":", ";", "i", "Y", "H", "#", "&"};

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
		for (int i = 1; i < input.length;) {

			// iterates through each line
			for (int j = 0; j < 28; j++, i++) {
				System.out.print(symbol[(int) Math.floor(Math.abs(input[i] - 0.01) * 10.0)]);
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