package com.dihel.backpropagation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StreamTokenizer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;
import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;
import javax.xml.bind.Unmarshaller;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlElementWrapper;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlTransient;

import com.basicClient.DriverInterface;
import com.basicClient.RaceConnector;


@XmlRootElement
public class BackpropagationNeuronNet {

	private int layersCount;
	
	private int isTrained = 0;

	private int inputsCount;

	private ArrayList<InputDescription> inputDescriptions;
	
	private int[] neuronInLayersCount;
	
	private ArrayList<String> outputDescriptions;
	
	private double learningRate;
	
	private double lastStepInfluenceLearningRate;
	
	private ArrayList<TrainSetElement> trainingSet;
	
	private ArrayList<TestSetElement> testSet;
	
	private String name;
	
	private File fileName;

	private ArrayList<Neuron> neurons;
	
	public void TestOnTraining() {
		// normalize inputs
		/*for (TrainSetElement element : this.trainingSet) {
			double[] trainInputs = element.getInputs();
			for (int i = 0; i < inputDescriptions.size(); i++) {
				trainInputs[i] = (trainInputs[i] - inputDescriptions.get(i).getMinimum())/(inputDescriptions.get(i).getMaximum() - inputDescriptions.get(i).getMinimum());  
			}
			element.setInputs(trainInputs);
		}*/
		
		double outputs[] = new double[neuronInLayersCount[layersCount-1]];
		for (TrainSetElement element : this.trainingSet) {
			try {
				
				outputs = forward(element.getInputs());
				
				System.out.println("inputs " + Arrays.toString(element.getInputs() ) + "outputs " + Arrays.toString(outputs));
			} catch (Exception e) {
				e.printStackTrace();
			}
			
		}
	}
	
	public void Test() 
	{
		// normalize inputs
		for (TestSetElement element : this.testSet) {
			double[] trainInputs = element.getInputs();
			for (int i = 0; i < inputDescriptions.size(); i++) {
				trainInputs[i] = (trainInputs[i] - inputDescriptions.get(i).getMinimum())/(inputDescriptions.get(i).getMaximum() - inputDescriptions.get(i).getMinimum());  
			}
			element.setInputs(trainInputs);
		}
		
		double outputs[] = new double[neuronInLayersCount[layersCount-1]];
		for (TestSetElement element : this.testSet) {
			try {
				
				outputs = forward(element.getInputs());
				
				System.out.println("inputs " + Arrays.toString(element.getInputs() ) + "outputs " + Arrays.toString(outputs));
			} catch (Exception e) {
				e.printStackTrace();
			}
			
		}
	}
	
	public void initNeurons() 
	{
		this.neurons = new ArrayList<Neuron>();
		for (int i = 0; i < inputsCount; i++) {
			this.neurons.add(new Neuron(1, ""));
		}
		
		for (int l = 0; l < layersCount; l++) {
			for (int i = 0; i < neuronInLayersCount[l]; i++) {
				int numberOfInputs = inputsCount;
				if (l > 0) {
					numberOfInputs = neuronInLayersCount[l-1];
				}
				this.neurons.add(new Neuron(numberOfInputs, ""));	
			}
		}
	}
	
	public double[] forward(double[] inputs) throws Exception {
		
		if (inputs.length != inputsCount) {
			throw new Exception("Inputs doesnt not match inputsCount");
		}
		
		for (int i = 0; i < inputsCount; i++) {
			neurons.get(i).setOutput(inputs[i]);
		}

		double sum = 0.0,output, y;
		int indexOffsetOfPreviousLayer, previousLayerCount;
		int indexOffset = inputsCount;
		double[] weigths;
		for (int l = 0; l < layersCount; l++) {
			//System.out.println("\n Layer " + l);
			for (int i = 0; i < neuronInLayersCount[l]; i++) {	
				sum = 0.0;
				
				indexOffsetOfPreviousLayer = indexOffset;
				previousLayerCount = 0;
				if (l-1 < 0) {
					previousLayerCount = inputsCount;
				} else {
					previousLayerCount = neuronInLayersCount[l-1];
				}
				indexOffsetOfPreviousLayer -= previousLayerCount;
				
				weigths = this.neurons.get(indexOffset + i).getWeights();
				sum += 1.0 * weigths[0]; 
				
				for (int j = 0; j < previousLayerCount; j++) {	
					output = neurons.get(indexOffsetOfPreviousLayer + j).getOutput();
					sum += weigths[j+1] * output;  
				}
				//System.out.println("Neuron " + i  + " Sum " + sum);
				y = 1.0/(1.0 + Math.exp(-1.0 * sum));
				//System.out.println("Neuron " + i  + " y " + y);
				this.neurons.get(indexOffset + i).setOutput(y);
			}
			indexOffset += neuronInLayersCount[l];
		}
		
		double outputs[] = new double[neuronInLayersCount[layersCount-1]];
		for (int i = 0; i < neuronInLayersCount[layersCount-1]; i++) {
			outputs[i] = neurons.get(neurons.size() - neuronInLayersCount[layersCount-1] + i).getOutput(); 
		}
		return outputs;
	}

	public double BackPropagation(double[] output, double[] expected) throws Exception {
		if (output.length != expected.length) {
			throw new Exception("Inputs doesnt not match inputsCount");
		}
		
		double error = 0.0;
		for (int i = 0; i < output.length; i++) {
			error += (expected[i] - output[i])*(expected[i] - output[i]);
		}

		error /= 2;
		
		
		int indexOffset = neurons.size();
		for (int l = layersCount-1; l >= 0; l--) {
			indexOffset -= neuronInLayersCount[l];
			for (int i = 0; i < neuronInLayersCount[l]; i++) {
				Neuron neuron = neurons.get(indexOffset+i);
				
				// last
				if (l == (layersCount-1)) {
					//double neuronError = (expected[i] - neuron.getOutput()) * (neuron.getOutput() * (1.0 - neuron.getOutput()));
					double neuronError = (neuron.getOutput() - expected[i]) * (neuron.getOutput() * (1.0 - neuron.getOutput()));
					neuron.setError(neuronError);
					
				} else {
					// j is index of neuron in next layer
					double d_error = 0.0;
					
					int indexOfNextLayer = indexOffset + neuronInLayersCount[l];
					
					for (int j = 0; j < neuronInLayersCount[l+1]; j++) {
						Neuron nextNeuron = neurons.get(indexOfNextLayer+j);
						d_error += (nextNeuron.getError() * nextNeuron.getWeights()[i+1]);
					}
					double neuronError = d_error * (neuron.getOutput() * (1.0 - neuron.getOutput()));
					neuron.setError(neuronError);
					
				}
				
				//double addition = this.learningRate * neuronError * neuron.getIntpu();
			}
		}
		
		indexOffset = neurons.size();
		// to 0, - inputs are -1

		for (int l = layersCount-1; l >= 0; l--) {
			int previousLayerCount = 0;
			if (l == 0) {
				previousLayerCount = inputsCount;
			} else {
				previousLayerCount = neuronInLayersCount[l-1];
			}
			
			indexOffset -= neuronInLayersCount[l];
			int previousNeuronIndex = indexOffset - previousLayerCount;
			
			for (int i = 0; i < neuronInLayersCount[l]; i++) {
				
				Neuron neuron = neurons.get(indexOffset+i);
				
				for (int j = 0; j < neuron.getWeights().length; j++) {

					double addition = 0.0;
					if (j > 0) {
						//Neuron previousNeuron = neurons.get(previousNeuronIndex+j-1);	
						Neuron previousNeuron = neurons.get(previousNeuronIndex+j-1);	
						//System.out.println("learningRate " + this.learningRate  + " error " + neuron.getError() + " output " + previousNeuron.getOutput());
						addition = -this.learningRate * neuron.getError() * previousNeuron.getOutput();
					} else {
						addition = -this.learningRate * neuron.getError() * 1;
					}
					//System.out.println("addition  " + addition + " error " + neuron.getError());
					
					//System.out.println("Neuron " + i  + " weig " + j + " addition " + addition);
					neuron.addToWeight(j, addition + neuron.getLastAdd() * this.lastStepInfluenceLearningRate);
					neuron.setLastAdd(addition);
				}
				
				/*for ( double a : neuron.getWeights()) {
					//System.out.println("Changed W  " + a);
				}*/
				
			}
		}
		
		return error;
	}
	
	public void setRandomWeights() {
		int indexOffset = this.inputsCount;
		Random rd = new Random();
		for (int l = layersCount-1; l >= 0; l--) {
			for (int i = 0; i < neuronInLayersCount[l]; i++) {
				Neuron neuron = neurons.get(indexOffset+i);
				double[] weights = new double[neuron.getWeights().length];
				weights[0] = 0.5;
				for (int j = 1; j < weights.length; j++) {
					weights[j] = rd.nextDouble();
				}
				neuron.setWeights(weights);
			}
			indexOffset += neuronInLayersCount[l];
		} 
	}
	
	public void setTestWeights() {
		
		
		int indexOffset = this.inputsCount;
		
		Neuron neuron = neurons.get(indexOffset+0);
		double[] weights = new double[neuron.getWeights().length];
		weights[0] = 0.5;
		weights[1] = 0.62;
		weights[2] = 0.55;
		neuron.setWeights(weights);
		
		neuron = neurons.get(indexOffset+1);
		weights = new double[neuron.getWeights().length];
		weights[0] = 0.5;
		weights[1] = 0.42;
		weights[2] = -0.17;
		neuron.setWeights(weights);
		
		neuron = neurons.get(indexOffset+2);
		weights = new double[neuron.getWeights().length];
		weights[0] = 0.5;
		weights[1] = 0.35;
		weights[2] = 0.81;
		neuron.setWeights(weights);
	}
	
	public void Train() {
		int epochCount = 5000000;
		double error = 0;
		
		// normalize inputs
		for (TrainSetElement element : this.trainingSet) {
			double[] trainInputs = element.getInputs();
			for (int i = 0; i < inputDescriptions.size(); i++) {
				trainInputs[i] = (trainInputs[i] - inputDescriptions.get(i).getMinimum())/(inputDescriptions.get(i).getMaximum() - inputDescriptions.get(i).getMinimum());  
			}
			element.setInputs(trainInputs);
		}
		
		System.out.println("Inputs " + inputsCount );
		System.out.println("LayersCount " + layersCount);
		int index = inputsCount;
		for (int l = 0; l < layersCount; l++) {
			double wei = 0;
			
			for (int i = 0; i < neuronInLayersCount[l]; i++) {
				wei += neurons.get(index+i).getWeights().length;
			}
			System.out.println("Layer " + l + " neuorons " + neuronInLayersCount[l] + " wei " + wei + " per neuron " + wei/neuronInLayersCount[l]);
			index += neuronInLayersCount[l];	
		}
		System.out.println("Neurons " + this.neurons.size());
		Scanner in = new Scanner(System.in); 
		String fileName = in.nextLine();
		// 
		setRandomWeights();
		//setTestWeights();
		
		double[] outputs = new double[this.neuronInLayersCount[this.layersCount-1]];
		double epochError = 0;
		for (int i = 0; i < epochCount; i++) {
			
			//System.out.println("\n Epoch  " + i);
			for (TrainSetElement element : this.trainingSet) {
				try {
					outputs = forward(element.getInputs());
					error = BackPropagation(outputs, element.getOutput());
					epochError += error;
				} catch (Exception e) {
					e.printStackTrace();
				}
				
			}
			if ((i % 500) == 0) {
				System.out.println("\n Epoch  " + i + " error " + epochError);
			}
			//System.out.println("Epoch " + i + " error " + error);
			if (epochError < 0.001) {
				System.out.println("DONE - breaking - " + i + " error " + error);
				break;
			}
			epochError = 0;
		}
		isTrained = 1;
	}
	
	@XmlElementWrapper(name="neurons")
	@XmlElement(name="neuron")
	public ArrayList<Neuron> getNeurons() {
		return neurons;
	}
	
	@XmlElement(name="layersCount")
	public int getLayersCount() {
		return layersCount;
	}

	public void setLayersCount(int layersCount) {
		this.layersCount = layersCount;
	}

	@XmlElement(name="inputsCount")
	public int getInputsCount() {
		return inputsCount;
	}
	

	@XmlElement(name="isTrained")
	public int getIsTrained() {
		return this.isTrained;
	}
	
	public void setInputsCount(int inputsCount) {
		this.inputsCount = inputsCount;
	}
	
	@XmlElementWrapper(name="inputDescriptions")
	@XmlElement(name="inputDescription")
	public ArrayList<InputDescription> getInputDescriptions() {
		return inputDescriptions;
	}
	
	public void setInputDescriptions(ArrayList<InputDescription> inputDescriptions) {
		this.inputDescriptions = inputDescriptions;
	}
	
	@XmlElementWrapper(name="neuronInLayersCount")
	@XmlElement(name="neuronInLayerCount")
	public int[] getNeuronInLayersCount() {
		return neuronInLayersCount;
	}
	
	public void setNeuronInLayersCount(int[] neuronInLayersCount) {
		this.neuronInLayersCount = neuronInLayersCount;
	}
	
	@XmlElementWrapper(name="outputDescriptions")
	@XmlElement(name="outputDescription")
	public ArrayList<String> getOutputDescriptions() {
		return outputDescriptions;
	}
	
	public void setOutputDescriptions(ArrayList<String> outputDescription) {
		this.outputDescriptions = outputDescription;
	}
	

	@XmlElement(name="learningRate")
	public double getLearningRate() {
		return learningRate;
	}
	
	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}
	

	@XmlElement(name="lastStepInfluenceLearningRate")
	public double getLastStepInfluenceLearningRate() {
		return lastStepInfluenceLearningRate;
	}
	
	public void setLastStepInfluenceLearningRate(
			double lastStepInfluenceLearningRate) {
		this.lastStepInfluenceLearningRate = lastStepInfluenceLearningRate;
	}
	
	@XmlElementWrapper(name="trainSet")
	@XmlElement(name="trainSetElement")
	public ArrayList<TrainSetElement> getTrainingSet() {
		return trainingSet;
	}
	
	public void setTrainingSet(ArrayList<TrainSetElement> trainingSet) {
		this.trainingSet = trainingSet;
	}
	
	@XmlElementWrapper(name="testSet")
	@XmlElement(name="testSetElement")
	public ArrayList<TestSetElement> getTestSet() {
		return testSet;
	}
	
	public void setTestSet(ArrayList<TestSetElement> testSet) {
		this.testSet = testSet;
	}
	
	@XmlTransient
	public String getName() {
		return name;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	@XmlTransient
	public File getFileName() {
		return fileName;
	}
	
	public void setFileName(File fileName) {
		this.fileName = fileName;
	}
	
	public BackpropagationNeuronNet(int layersCount, int inputsCount,
			List<InputDescription> inputDescriptions, int[] neuronInLayersCount,
			List<String> outputDescription, double learningRate,
			double lastStepInfluenceLearningRate,
			List<TrainSetElement> trainingSet, List<TestSetElement> testSet,
			List<Neuron> neurons
			) {
		super();
		this.layersCount = layersCount;
		this.inputsCount = inputsCount;
		this.inputDescriptions = new ArrayList<>(inputDescriptions);
		this.neuronInLayersCount = neuronInLayersCount;
		this.outputDescriptions = new ArrayList<>(outputDescription);
		this.learningRate = learningRate;
		this.lastStepInfluenceLearningRate = lastStepInfluenceLearningRate;
		if(trainingSet != null){
			this.trainingSet = new ArrayList<>(trainingSet);
		}
		if(testSet != null){
			this.testSet = new ArrayList<>(testSet);
		}
		if(neurons != null){
			this.neurons = new ArrayList<>(neurons);
		} else {
			this.neurons = new ArrayList<>();
		}
	}
	
	public BackpropagationNeuronNet() {
		this(3, 2, 
				Arrays.asList(
						new InputDescription("i1", 0, 1),
						new InputDescription("i2", 0, 1)
						),
				new int[]{5,3,1},
				Arrays.asList("output"),
				0.4,
				0.1,
				new ArrayList<TrainSetElement>(),
				new ArrayList<TestSetElement>(),
				new ArrayList<Neuron>()
				);
	}

	public static BackpropagationNeuronNet readFromXml(Reader input) throws JAXBException{
		JAXBContext context = JAXBContext.newInstance(BackpropagationNeuronNet.class);
		Unmarshaller m =  context.createUnmarshaller();
		return (BackpropagationNeuronNet)m.unmarshal(input);

	}

	public static BackpropagationNeuronNet readFromTxt(Reader input) throws IOException{
		StreamTokenizer streamTokenizer = new StreamTokenizer(input);
		streamTokenizer.commentChar('#');

		streamTokenizer.nextToken();
		int layers = (int) streamTokenizer.nval;

		streamTokenizer.nextToken();
		int inputsCount = (int) streamTokenizer.nval;

		ArrayList<InputDescription> inputDescriptions = new ArrayList<>();
		//		String[] inputsNames = new String[inputsCount];
		//		double[] inputsMin = new double[inputsCount];
		//		double[] inputsMax = new double[inputsCount];
		for (int i = 0; i < inputsCount; i++) {
			streamTokenizer.nextToken();
			String inputsName = streamTokenizer.sval;
			streamTokenizer.nextToken();
			double inputMin = streamTokenizer.nval;
			streamTokenizer.nextToken();
			double inputMax = streamTokenizer.nval;
			inputDescriptions.add(new InputDescription(inputsName, inputMin, inputMax));
		}
		int[] neuronsCount = new int[layers];
		for (int i = 0; i < layers; i++) {
			streamTokenizer.nextToken();
			neuronsCount[i] = (int) streamTokenizer.nval;
		}
		String[] outputNames = new String[neuronsCount[neuronsCount.length - 1]];
		for (int i = 0; i < outputNames.length; i++) {
			streamTokenizer.nextToken();
			outputNames[i] = streamTokenizer.sval;
		}
		streamTokenizer.nextToken();
		double learningRate = streamTokenizer.nval;

		streamTokenizer.nextToken();
		double lastStepLearningRate = streamTokenizer.nval;

		streamTokenizer.nextToken();
		int numberOfElementsInTrainSet = (int) streamTokenizer.nval;
		ArrayList<TrainSetElement> trainSet = new ArrayList<>();
		for (int i = 0; i < numberOfElementsInTrainSet; i++) {
			double[] elementUserInputs = new double[inputsCount];
			double[] elementOutputs = new double[outputNames.length];
			for (int j = 0; j < inputsCount; j++) {
				streamTokenizer.nextToken();
				double value = streamTokenizer.nval;
				elementUserInputs[j] = value;
			}
			for (int k = 0; k < outputNames.length; k++) {
				streamTokenizer.nextToken();
				double value = streamTokenizer.nval;
				elementOutputs[k] = value;
			}
			trainSet.add(new TrainSetElement(elementUserInputs, elementOutputs));
		}

		streamTokenizer.nextToken();
		int numberOfElementsInTestSet = (int) streamTokenizer.nval;
		ArrayList<TestSetElement> testSet = new ArrayList<>();
		for (int i = 0; i < numberOfElementsInTrainSet; i++) {
			double[] elementUserInputs = new double[inputsCount];
			double[] elementOutputs = new double[outputNames.length];
			for (int j = 0; j < inputsCount; j++) {
				streamTokenizer.nextToken();
				double value = streamTokenizer.nval;
				elementUserInputs[j] = value;
			}
			testSet.add(new TestSetElement(elementUserInputs));
		}

		BackpropagationNeuronNet netConfig = new BackpropagationNeuronNet(
				layers, inputsCount, inputDescriptions, neuronsCount, 
				Arrays.asList(outputNames), learningRate, lastStepLearningRate, 
				trainSet, testSet, new ArrayList<Neuron>()
				
				);

		return netConfig;
	}

	public static BackpropagationNeuronNet load() {
		JFileChooser d = new JFileChooser();
		d.setDialogTitle("Load net config");
		d.setDialogType(JFileChooser.OPEN_DIALOG);
		d.setFileFilter(new javax.swing.filechooser.FileFilter() {
			public boolean accept(File f) {
				if (f.isDirectory()) {
					return true;
				}
				if (f.getName().endsWith(".txt")) {
					return true;
				}
				return false;
			}

			public String getDescription() {
				return "Neuron Net Description files";
			}
		});
		if (d.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
			if (d.getSelectedFile().exists()) {
				FileReader fileReader = null;
				try {
					fileReader = new FileReader(d.getSelectedFile());
				} catch (FileNotFoundException e2) {
					e2.printStackTrace();
				}
				try {
					String netName = d.getSelectedFile().getName().substring(0, d.getSelectedFile().getName().lastIndexOf('.'));
					BackpropagationNeuronNet netConfig = readFromTxt(fileReader);
					netConfig.setName(netName);
					netConfig.setFileName(d.getSelectedFile());
					fileReader.close();
					return netConfig;
				} catch (IOException e1) {
					e1.printStackTrace();
				}

			} else{
				JOptionPane.showMessageDialog(null, "File " + d.getSelectedFile().getName() + "don't exists.", "Chyba", JOptionPane.OK_OPTION);
			}
		}
		return null;
	}

	public void storeToXML(File file) throws JAXBException{
		JAXBContext context = JAXBContext.newInstance(BackpropagationNeuronNet.class);
		Marshaller m =  context.createMarshaller();
		m.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true);
		if(file == null){
			file = Paths.get(getFileName().getParentFile().getAbsolutePath(), getName()+".xml").toFile();
		}
		m.marshal(this, file);
		//m.marshal(this, System.out);
	}

	public void mainLoad() {
		try {
			load().storeToXML(null);
		} catch (JAXBException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	public void runRace(String host, int port, String raceName, String driverName, String carType) throws Exception {
        RaceConnector raceConnector = null;
        raceConnector = new RaceConnector(host, port, null);
        List<String> raceList = raceConnector.listRaces();
        List<String> carList = raceConnector.listCars(raceName);
        raceConnector.setDriver(new DriverInterface() {

            @Override
            public HashMap<String, Float> drive(HashMap<String, Float> values) {
                HashMap<String, Float> responses = new HashMap<String, Float>();
                double[] results = new double[2];
                try {
                	
                	double[] inputs = new double[values.size()];
                	int i = 0;
                	for (Map.Entry<String, Float> entry : values.entrySet()) {
                		//System.out.println("String " + entry.getKey());
                		//System.out.println("Value " + entry.getValue());
                		inputs[i] = entry.getValue();
                		System.out.println("Input " + i + " calue: " + inputs[i]);
                		i++;
                	}
                	//System.out.println("Inputs " + Arrays.toString(inputs));
                    results = forward(inputs);

                	System.out.println("results " + Arrays.toString(results));
                } catch (Exception e) {
                    e.printStackTrace();
                }
                //results[1] = results[1] > results[0] ? results[1] : 0;

                //results[0] = results[0] > results[1] ? results[0] : 0;
                responses.put("wheel", (float) results[1]);
                responses.put("acc", (float) results[0]);
                return responses;
            }
        });
        raceConnector.start(raceName, driverName, carType);
    }
}
