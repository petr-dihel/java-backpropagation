package com.dihel.backpropagation;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;

public class TrainSetElement extends TestSetElement {

	private double[] output;

	public TrainSetElement() {
		this(2, 1);
	}

	public TrainSetElement(int inputsSize, int outputSize) {
		super(inputsSize);
		output = new double[outputSize];
	}

	@XmlElementWrapper(name="outputs")
	@XmlElement(name="value")
	public double[] getOutput() {
		return output;
	}

	public void setOutput(double[] output) {
		this.output = output;
	}

	public TrainSetElement(double[] input, double[] output) {
		super(input);
		this.output = output;
	}

}
