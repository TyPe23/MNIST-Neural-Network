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

// main class
public class Assignment2 {

	// start here
	public static void main(String[] args) throws Exception{
		System.out.println("Hello world!");

		// create a new file from the .csv training file
		Scanner sc = new Scanner(new File(System.getProperty("user.dir") + "\\mnist_train.csv"));

		// set the delimiter to a comma
		sc.useDelimiter(",");

		// print out all of the values to make sure it works
		int i = 0;

		while (i < 783 * 3 + 4) {
			if (sc.hasNext()) {
				System.out.println(sc.next());
			}
			i++;
		}

		// close the .csv file before exiting
		sc.close();
	}
}