package com.dihel.backpropagation;

public class InputDescription {

	private String name;
	private double minimum;
	private double maximum;
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public double getMinimum() {
		return minimum;
	}
	public void setMinimum(double minimum) {
		this.minimum = minimum;
	}
	public double getMaximum() {
		return maximum;
	}
	public void setMaximum(double maximum) {
		this.maximum = maximum;
	}
	
	public double recalculateToNeuron(double value){
		return (value - minimum) / (maximum - minimum);
	}

	public double recalculateToNaturalRange(double value){
		return value * (maximum - minimum) + minimum;
	}
	
	public InputDescription() {
		this("", 0, 1);
	}

	public InputDescription(String description) {
		this(description, 0, 1);
	}
	public InputDescription(String name, double minimum, double maximum) {
		super();
		this.name = name;
		this.minimum = minimum;
		this.maximum = maximum;
	}
	@Override
	public String toString() {
		return name + " (" + minimum
				+ ", " + maximum + ")";
	}

	
}
