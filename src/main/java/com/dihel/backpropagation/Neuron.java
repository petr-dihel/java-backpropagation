package com.dihel.backpropagation;

import java.util.Arrays;
import java.util.List;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;


public class Neuron {
	
	private double[] weights;
	
	private double[] inputs;
	
	private double output;
	
	private double error;
	
	private double lastAdd = 0.0;
	
	@XmlElement(name="error")
	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = error;
	}

	private String name = "y";
	
	public Neuron() {
		//this(2);
	}

	public Neuron(
			int numberOfInputs,
			String name
	) 
	{
		System.out.println("Number of inputs " + (numberOfInputs + 1));
		weights = new double[(1+numberOfInputs)];
		inputs = new double[(numberOfInputs+1)];
		
	}
	
	@XmlElement(name="name")
	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	@XmlElementWrapper(name="weights")
	@XmlElement(name="weight")
	public double[] getWeights() {
		return weights;
	}
	
	public void addToWeight(int index, double addition) {
		this.weights[index] += addition;
	}

	public void setWeights(double[] weights) {
		this.weights = weights;
	}
	
	public void setInputs(double[] inputs) {
		this.inputs = inputs;
	}
	
	@XmlElementWrapper(name="inputs")
	@XmlElement(name="value")
	public double[] getInputs() {
		return this.inputs;
	}

	public void setOutput(double output) {
		this.output = output;
	}
	
	@XmlElement(name="output")
	public double getOutput() {
		return this.output;
	}
	
	@Override
	public String toString() {
		return "Perceptron [weights=" + Arrays.toString(weights) + ", inputs="
				+ Arrays.toString(inputs) + ", name=" + name + "]";
	}
	
	public void setLastAdd(double add) {
		this.lastAdd = add;
	}
	
	public double getLastAdd() {
		return this.lastAdd;
	}
	
}
