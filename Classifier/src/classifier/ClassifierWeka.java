/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.core.converters.ConverterUtils.*;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Vincent
 */
public class ClassifierWeka {
    
    public static BufferedReader input = new BufferedReader(new InputStreamReader(System.in));
    
    public static Instances loadData(String filename) throws Exception {
        DataSource source = new DataSource("data/"+ filename);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        
        System.out.println("Successfully Load Data");
        
        return data;
    }
    
    public static Instances removeAttribute(Instances data, String index) throws Exception{
        String[] options = new String[2];
        options[0] = "-R";                                    // "range"
        options[1] = index;                                   // attribute
        Remove remove = new Remove();                         // new instance of filter
        remove.setOptions(options);                           // set options
        remove.setInputFormat(data);                          // inform filter about dataset **AFTER** setting options
        Instances newData = Filter.useFilter(data, remove);   // apply filter
        
        System.out.println("Successfully Remove Attribute");
        
        return newData;
    }
    
    public static Instances resample(Instances data){
        final Resample filter = new Resample();
	Instances filteredInstances = null;
	filter.setBiasToUniformClass(1.0);
	try {
		filter.setInputFormat(data);
		filter.setNoReplacement(false);
		filter.setSampleSizePercent(100);
		filteredInstances = Filter.useFilter(data, filter);
	} catch (Exception e) {
		System.out.println("Error when resampling input data!");
		e.printStackTrace();
	}
        
        System.out.println("Successfully Resample");
        
	return filteredInstances;
    }
    
    public static void buildClassifier(Classifier classifierModel, Instances data) throws Exception {
        classifierModel.buildClassifier(data);
        
        System.out.println("Successfully Build Classifier");
    }
    
    public static void evaluateModel(Classifier classifierModel, Instances trainData, Instances testData) throws Exception{
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(classifierModel, testData);
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
    }
    
    public static void crossValidation(Classifier classifierModel, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifierModel, data, 10, new Random(1));
        System.out.println("Accuracy: "+Double.toString(eval.pctCorrect()));
    }
    
    public static void percentageSplit(Classifier classifierModel, Instances data) throws IOException, Exception{
        System.out.print("Split percentage = ");
        double splitPercentage = Double.parseDouble(input.readLine())/100;
        data.randomize(new java.util.Random(0));
        int trainSize = (int) Math.round(data.numInstances() * splitPercentage);
        int testSize = data.numInstances() - trainSize;
        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);
        evaluateModel(classifierModel, trainData, testData);
    }
    
    public static void saveModel(Classifier classifierModel, String filename) throws Exception {
        ObjectOutputStream oos = null;
        try {
            oos = new ObjectOutputStream(new FileOutputStream("models/" + filename + ".model"));
        } catch (FileNotFoundException e1) {
            e1.printStackTrace();
        } catch (IOException e1) {
            e1.printStackTrace();
        }
        oos.writeObject(classifierModel);
        oos.flush();
        oos.close();
        
        System.out.println("Successfully Save Model");
    }

    public static Classifier loadModel(String filename) throws Exception {
        Classifier classifierModel;

        FileInputStream fis = new FileInputStream("models/" + filename + ".model");
        ObjectInputStream ois = new ObjectInputStream(fis);

        classifierModel = (Classifier) ois.readObject();
        ois.close();
        
        System.out.println("Successfully Load Model");

        return classifierModel;
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException, Exception {
        Classifier classifierModel = null;
        
        System.out.print("Data filename : ");
        String filename = input.readLine();
        Instances data = loadData(filename);
        
        int classifierMenu = 999;
        int option = 999;
        while(classifierMenu != 0){
            System.out.println("\nChoose Classifier\n=================\n");
            System.out.println("1. Id3 Weka Classifier");
            System.out.println("2. J48 Weka Classifier");
            System.out.println("3. MyId3 Classifier");
            System.out.println("4. MyJ48 Classifier");
            System.out.println("0. Exit\n");
            
            System.out.print("Choose Classifier : ");
            classifierMenu = Integer.parseInt(input.readLine());
            System.out.println();
            switch(classifierMenu) {
                case 0 :
                    System.out.println("Bye-bye!");
                    option = 0;
                    break;
                case 1 :
                    classifierModel = (Classifier)new Id3();
                    buildClassifier(classifierModel, data);
                    System.out.println("Id3 Weka classifier has been built");
                    option = 999;
                    break;
                case 2 :
                    classifierModel = (Classifier)new J48();
                    buildClassifier(classifierModel, data);
                    System.out.println("J48 Weka classifier has been built");
                    option = 999;
                    break;
                case 3 :
                    classifierModel = (Classifier)new Id3();
                    buildClassifier(classifierModel, data);
                    System.out.println("MyId3 classifier has been built");
                    option = 999;
                    break;
                case 4 : 
                    classifierModel = (Classifier)new J48();
                    buildClassifier(classifierModel, data);
                    System.out.println("MyJ48 classifier has been built");
                    option = 999;
                    break;
                default :
                   System.out.println("Invalid Input");
                   option = 0;
                   break;
            }
            
            while (option != 0) {
                System.out.println();
                System.out.println("\nChoose Menu\n=================\n");
                System.out.println("1. Remove Attribute");
                System.out.println("2. Filter : Resample");
                System.out.println("3. Testing Model Given Data Set");
                System.out.println("4. 10-fold Cross Validation");
                System.out.println("5. Percentage Split");
                System.out.println("6. Save Model");
                System.out.println("7. Load Model");
                System.out.println("8. Classify Unseen Data");
                System.out.println("9. Change Classifier");
                System.out.println("0. Exit\n");
                
                System.out.print("Choose Menu : ");
                option = Integer.parseInt(input.readLine());
                System.out.println();
                switch(option) {
                    case 0 :
                        System.out.println("Bye-bye!");
                        option = 0;
                        classifierMenu = 0;
                        break;
                    case 1 :
                        System.out.print("Attribute column to remove : ");
                        String attribute = input.readLine();
                        data = removeAttribute(data, attribute);
                        System.out.println(data.toString());
                        break;
                    case 2 :
                        data = resample(data);
                        System.out.println("Data after resample :");
                        System.out.println(data.toString());
                        break;
                    case 3 :
                        System.out.print("Test data filename : ");
                        filename = input.readLine();
                        Instances testData = loadData(filename);
                        evaluateModel(classifierModel,data,testData);
                        break;
                    case 4 :
                        crossValidation(classifierModel,data);
                        break;
                    case 5 :
                        percentageSplit(classifierModel,data);
                        break;
                    case 6 :
                        System.out.print("Save model filename : ");
                        filename = input.readLine();
                        saveModel(classifierModel, filename);
                        break;
                    case 7 :
                        System.out.print("Load model filename : ");
                        filename = input.readLine();
                        classifierModel = loadModel(filename);
                        break;
                    case 8 : 
                        Instances unseenData = loadData("unseen-data.arff");

                        System.out.println("Classify Result : ");
                        for (int i=0; i < unseenData.numInstances(); i++) {
                            double clsLabel = classifierModel.classifyInstance(unseenData.instance(i));
                            System.out.println(unseenData.classAttribute().value((int) clsLabel));
                        }
                        break;
                    case 9 :
                        option = 0;
                        break;
                    default :
                       System.out.println("Invalid Input");
                       option = 0;
                       classifierMenu = 0;
                       break;
                }
            }
        }
    }
    
}
