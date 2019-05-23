package net;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import org.supercsv.io.CsvListReader;
import org.supercsv.prefs.CsvPreference;

public class Network {

	private ArrayList<ArrayList<Double>> weights = new ArrayList<ArrayList<Double>>();
	private ArrayList<ArrayList<Double>> vals = new ArrayList<ArrayList<Double>>();
	private ArrayList<ArrayList<Double>> deltas = new ArrayList<ArrayList<Double>>();

	private ArrayList<Double> inputs = new ArrayList<>();
	private ArrayList<Double> targets = new ArrayList<>();
	private ArrayList<Double> outputs = new ArrayList<>();

	private ArrayList<ArrayList<Double>> trainingData = new ArrayList<ArrayList<Double>>();
	private ArrayList<Double> pairedTargets = new ArrayList<>();

	private int numLayers;
	Scanner scnr = new Scanner(System.in);

	public static void main(String[] args) throws IOException {

		ArrayList<Double> userInputs = new ArrayList<Double>();

		Network nn = new Network(3, userInputs);
		nn.parseCsv();

		nn.setInputLayer(0);
		nn.setHiddenLayerSizes();
		nn.setTargets(0);
		nn.forwardPropagate();
		nn.backPropagate();

		for (int i = 1; i < 1000; i++) {
			nn.setInputLayer(i);
			nn.setHiddenLayerSizes();
			nn.setTargets(i);
			nn.forwardPropagate();
			nn.backPropagate();
			// System.out.println(nn.vals.get(nn.numLayers - 1).get(0) + "new predicted
			// output");
			// System.out.println(nn.vals.get(nn.numLayers - 1).get(1) + "new predicted
			// output");
		}
		System.out.println("-----Ready to Predict-----");
		while (true) {
			if (nn.scnr.next().charAt(0) == 'q') {
				break;
			}
			nn.predict();
		}

		nn.scnr.close();
	}

	// this using arrayList for different sized hidden layers, flexible
	public Network(int layers, ArrayList<Double> input) {
		for (int i = 0; i < layers; i++) {
			vals.add(new ArrayList<Double>());
		}
		for (int i = 0; i < layers - 1; i++) {
			weights.add(new ArrayList<Double>());
			deltas.add(new ArrayList<Double>());
		}

		this.numLayers = layers;
		this.inputs = input;
	}

	public void predict() {
		Random rand = new Random();
		int curIndex = rand.nextInt(300);
		setInputLayer(curIndex);
		setHiddenLayerSizes();
		setTargets(curIndex);
		forwardPropagate();
		predictOutcome();
	}

	public void setInputLayer(int index) {
		if (index == 0) {
			for (int i = 0; i < trainingData.get(index).size(); i++) {
				vals.get(0).add(0.0);
			}
		}
		for (int i = 0; i < trainingData.get(index).size(); i++) {
			vals.get(0).set(i, trainingData.get(index).get(i));
			// System.out.println(vals.get(0).get(i));
		}
	}

	public void setTargets(int index) {
		if (index == 0) {
			for (int i = 0; i < 10; i++) {
				targets.add(0.0);
			}
		}
		for (int i = 0; i < 10; i++) {
			// double target = scnr.nextDouble();
			if (i == pairedTargets.get(index)) {
				targets.set(i, pairedTargets.get(index));
			} else {
				targets.set(i, 0.0);
			}
			System.out.println(targets.get(i));
		}
	}

	public void setHiddenLayerSizes() {
		Random rand = new Random(5);

		for (int i = 1; i < this.numLayers; i++) { // setting hidden val sizes
			// System.out.println("Input size");
			// int size = scnr.nextInt();
			vals.get(i).clear();
			for (int j = 0; j < 50 - 20 * i; j++) {
				vals.get(i).add(0.0);

			}
			for (int k = 0; k < vals.get(i - 1).size() * vals.get(i).size(); k++) {
				weights.get(i - 1).add(rand.nextDouble());
				deltas.get(i - 1).add(0.0);
			}
			// System.out.println(vals.get(i - 1).size() * vals.get(i).size());

		}
	}

	public void forwardPropagate() {
		for (int i = 1; i < numLayers; i++) {
			double sum = 0;
			// System.out.println(" ");
			for (int j = 0; j < vals.get(i).size(); j++) {
				for (int h = 0; h < vals.get(i - 1).size(); h++) {
					sum += weights.get(i - 1).get(j * vals.get(i - 1).size() + h) * vals.get(i - 1).get(h);

				}
				// System.out.println(sum + " this is the sum for" + i);
				vals.get(i).set(j, 1.0 / (1.0 + Math.pow(Math.E, -sum / 100)));
				// System.out.println(" ");
				// System.out.println(vals.get(i).get(j) + " = neuron(" + i + ")(" + j + ")");
				sum = 0;
			}
		}
	}

	public void backPropagate() {
		for (int i = numLayers - 1; i > 0; i--) {
			if (i == numLayers - 1) {
				for (int j = 0; j < vals.get(i - 1).size(); j++) {
					for (int k = 0; k < vals.get(i).size(); k++) {
						deltas.get(i - 1).set(j + k * vals.get(i - 1).size(),
								deltas.get(i - 1).get(j + k * vals.get(i - 1).size())
										+ outputErrorDeriv(vals.get(numLayers - 1).get(k), targets.get(k))
												* sigmoidDeriv(vals.get(i - 1).get(j)) * vals.get(i - 1).get(j));
						weights.get(i - 1).set(j + k * vals.get(i - 1).size(),
								weights.get(i - 1).get(j + k * vals.get(i - 1).size())
										- 0.5 * deltas.get(i - 1).get(j + k * vals.get(i - 1).size()));
						// System.out.println("New w" + i + "" + "(" + (j + k * vals.get(i - 1).size())
						// + ")" + " is: "
						// + weights.get(i - 1).get(j + k * vals.get(i - 1).size()));
					}
				}
			} else {
				for (int j = 0; j < vals.get(i - 1).size(); j++) {
					for (int k = 0; k < vals.get(i).size(); k++) {
						deltas.get(i - 1).set(j + k * vals.get(i - 1).size(),
								deltas.get(i - 1).get(j + k * vals.get(i - 1).size())
										* sigmoidDeriv(vals.get(i - 1).get(j)) * vals.get(i - 1).get(j));
						weights.get(i - 1).set(j + k * vals.get(i - 1).size(),
								weights.get(i - 1).get(j + k * vals.get(i - 1).size())
										- 0.5 * deltas.get(i - 1).get(j + k * vals.get(i - 1).size()));
						// System.out.println("New w" + i + "" + "(" + (j + k * vals.get(i - 1).size())
						// + ")" + " is: "
						// + weights.get(i - 1).get(j + k * vals.get(i - 1).size()));
					}
				}
			}
		}
	}

	public double calculateAbsErr() {
		double sum = 0;

		for (int i = 0; i < vals.get(numLayers - 1).size(); i++) {
			sum += 0.5 * Math.pow(targets.get(i) - vals.get(numLayers - 1).get(i), 2);
		}
		return sum;
	}

	public void predictOutcome() {

		ArrayList<Double> listResults = new ArrayList<Double>();
		for (int i = 0; i < 10; i++) {
			listResults.add(0.5 * Math.pow(targets.get(i) - vals.get(numLayers - 1).get(i), 2));
		}
		// System.out.println(listResults);
		double min = Collections.max(listResults);

		for (int i = 0; i < listResults.size(); i++) {
			if (min == listResults.get(i)) {
				System.out.println(i);
			}
		}

		listResults.clear();

	}

	public double outputErrorDeriv(double val, double target) {
		return -(target - val);
	}

	public double sigmoidDeriv(double output) {

		return output * (1 - output);
	}

	public ArrayList<ArrayList<Double>> parseCsv() throws IOException {
		String

		csvFile = "C:/Users/wanga/eclipse-workspace1/SimpleNN/MNIST/mnist_train.csv";
		FileReader fr = new FileReader(csvFile);

		CsvListReader cs = new CsvListReader(fr, CsvPreference.STANDARD_PREFERENCE);

		int count;

		try {
			count = 0;
			boolean isLabel;
			while (count < 1000) {
				trainingData.add(new ArrayList<Double>());
				isLabel = true;
				for (String s : cs.read()) {
					if (isLabel) {
						trainingData.get(count).add(Double.valueOf(s)); //
						// System.out.print(Double.valueOf(s));
					} else {
						if (Double.valueOf(s) != 0) {
							trainingData.get(count).add(1.0); //
							// System.out.print(1.0);
						} else {
							trainingData.get(count).add(0.0); //
							// System.out.print(0.0);
						}
					}
					isLabel = false;
				}
				// System.out.println("");
				count++;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		for (ArrayList<Double> d : trainingData) {
			pairedTargets.add(d.get(0));
			d.remove(0);
		}
		cs.close();
		return trainingData;
	}
}
