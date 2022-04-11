package com.dihel.backpropagation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.nio.file.Paths;
import java.util.Scanner;

import javax.xml.bind.JAXBException;

public class App {

	public static void Train() 
	{
		System.out.println("Enter xml file with ");
		Scanner in = new Scanner(System.in); 
		String fileName = in.nextLine();
		try {
			FileReader fileReader = new FileReader(fileName);
			BackpropagationNeuronNet net = BackpropagationNeuronNet.readFromXml(fileReader);
			net.initNeurons();
			net.Train();
			
			System.out.println("Save xml file ? \n 0 - dont \n 1 - save default \n 2 - enter file name \n");
			String s = in.nextLine();
			int save= Integer.parseInt(s);
			if (save > 0) {
				if (save == 2) {
					System.out.println("enter file name");
					fileName = in.nextLine();
				} else {
					fileName = fileName.substring(0, fileName.length()-4) + "_trained.xml";
				}
				File file = new File(fileName);
				net.storeToXML(file);
				System.out.println("Saved to " + fileName);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void Test() 
	{
		System.out.println("Test Enter xml file with ");
		Scanner in = new Scanner(System.in); 
		String fileName = in.nextLine();
		try {
			FileReader fileReader = new FileReader(fileName);
			BackpropagationNeuronNet net = BackpropagationNeuronNet.readFromXml(fileReader);
			if (net.getIsTrained() == 1) {
				throw new Exception("Not trained " + net.getIsTrained() );
			}
			//net.initNeurons();
			net.TestOnTraining();
			net.Test();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void Race() 
	{
		System.out.println("Test Enter xml file with 0 - default");
		Scanner in = new Scanner(System.in); 
		String fileName = in.nextLine();
		try {
			if (Integer.parseInt(fileName) == 0) {
				fileName = "car_prosim_trained.xml";
			}
			FileReader fileReader = new FileReader(fileName);
			BackpropagationNeuronNet net = BackpropagationNeuronNet.readFromXml(fileReader);
			if (net.getIsTrained() == 1) {
				throw new Exception("Not trained " + net.getIsTrained() );
			}
			//java.cs.vsb.cz
			//net.runRace("java.cs.vsb.cz", 9460, "Race", "dih0008", null);
			net.runRace("localhost", 9461, "test", "dih0008", null);
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		
		FileReader fileReader;
		try {
			System.out.println("Mode: \n 1 - Train \n 2 - Test \n 3 - Race \n");
			Scanner in = new Scanner(System.in); 
			String s = in.nextLine();
			int mode = Integer.parseInt(s);
			
			switch (mode) {
			case 1:
				Train();
				break;
			case 2:
				Test();
				break;
			case 3:
				Race();
				break;
			default:
				System.out.println("Wrong mode");
				break;
			}
			 
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("END");

	}
}
