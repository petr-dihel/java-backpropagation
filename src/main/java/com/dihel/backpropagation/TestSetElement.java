package com.dihel.backpropagation;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;

public class TestSetElement {

	private double[] inputs;
	
	
	public TestSetElement() {
		this(2);
	}

	public TestSetElement(int inputsSize){
		inputs = new double[inputsSize];
	}

	@XmlElementWrapper(name="inputs")
	@XmlElement(name="value")
	public double[] getInputs() {
		return inputs;
	}

	public void setInputs(double[] inputs) {
		this.inputs = inputs;
	}

	public TestSetElement(double[] inputs) {
		super();
		this.inputs = inputs;
	}
}
