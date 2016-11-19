package J48;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class Util {

    private static String pathDataSet = "dataSet/";
    private static String pathSavedModel = "savedModel/";
    private static String pathClassifyResult = "classifiedInstance/";

    public static Instances readARFF(String namaFile)
    {
        try
        {
            Instances dataSet;
            ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(pathDataSet + namaFile);
            dataSet = dataSource.getDataSet();
            if(dataSet.classIndex() == -1)
            {
                dataSet.setClassIndex(dataSet.numAttributes()-1);
            }
            return dataSet;
        }

        catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    public static Instances readCSV(String namaFile)
    {
        try
        {
            CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(new File(pathDataSet + namaFile));
            Instances dataSet = csvLoader.getDataSet();
            if(dataSet.classIndex() == -1)
            {
                dataSet.setClassIndex(dataSet.numAttributes()-1);
            }
            return dataSet;
        }

        catch (IOException e)
        {
            e.printStackTrace();
            return null;
        }
    }

    public static Instances removeAttribute(Instances dataSet, int attributeIndex)
    {
        try
        {
            String options[] = new String[2];
            options[0] = "-R";
            options[1] = String.valueOf(attributeIndex);
            Remove remove = new Remove();
            remove.setOptions(options);
            remove.setInputFormat(dataSet);
            Instances newDataSet = Filter.useFilter(dataSet,remove);
            return newDataSet;
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    public static Instances resampleDataSet(Instances dataSet)
    {
        try
        {
            Resample resample = new Resample();
            String filterOptions = "-B 0.0 -S 1 -Z 100.0";
            resample.setOptions(Utils.splitOptions(filterOptions));
            resample.setRandomSeed((int) System.currentTimeMillis());
            resample.setInputFormat(dataSet);
            Instances newDataSet = Filter.useFilter(dataSet,resample);
            return newDataSet;
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    public static Classifier buildClassifier(Instances dataSet, Classifier classifier)
    {
        try {
            classifier.buildClassifier(dataSet);
            return classifier;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static Evaluation testClassifier(Classifier classifier, Instances dataSet, Instances testSet)
    {
        try
        {
            Evaluation evaluation = new Evaluation(dataSet);
            evaluation.evaluateModel(classifier, testSet);
            return evaluation;
        }

        catch (Exception e)
        {
            e.printStackTrace();
        }
        return null;
    }

    public static Evaluation crossValidationTest(Instances dataSet, Classifier untrainedClassifier)
    {
        try
        {
            Evaluation eval = new Evaluation(dataSet);
            eval.crossValidateModel(untrainedClassifier, dataSet, 10, new Random(1));
            return eval;
        }

        catch (Exception e)
        {
            e.printStackTrace();
        }
        return null;
    }

    public static Evaluation percentageSplit(Instances dataSet, Classifier untrainedClassifier, int percentage)
    {
        Instances data = new Instances(dataSet);
        data.randomize(new Random(1));

        int trainSize = Math.round(data.numInstances() * percentage / 100);
        int testSize = data.numInstances() - trainSize;
        Instances trainSet = new Instances(data, 0, trainSize);
        Instances testSet = new Instances(data, trainSize, testSize);

        try
        {
            untrainedClassifier.buildClassifier(trainSet);
            Evaluation eval = testClassifier(untrainedClassifier, trainSet, testSet);
            return eval;
        }

        catch (Exception e)
        {
            e.printStackTrace();
        }
        return null;
    }

    public static void saveModel(String filename, Classifier classifier)
    {
        try
        {
            SerializationHelper.write(pathSavedModel + filename, classifier);
        }

        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    public static Classifier loadModel(String filename)
    {
        try
        {
            return (Classifier) SerializationHelper.read(pathSavedModel + filename);
        }

        catch (Exception e)
        {
            e.printStackTrace();
        }
        return null;
    }

    public static void classify(String filename, Classifier classifier)
    {
        try
        {
            Instances input = readARFF(filename);
            input.setClassIndex(input.numAttributes()-1);
            for(int i=0; i<input.numInstances(); i++)
            {
                double classLabel = classifier.classifyInstance(input.instance(i));
                input.instance(i).setClassValue(classLabel);
                System.out.println("Instance: " + input.instance(i));
                System.out.println("Class: " + input.classAttribute().value((int)classLabel));
            }

            BufferedWriter writer = new BufferedWriter(
            new FileWriter(pathClassifyResult + "labeled." + filename));
            writer.write(input.toString());
            writer.newLine();
            writer.flush();
            writer.close();
        }

        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    public static void main(String [] args)
    {
        System.out.println("========== Reading File From ARFF ==========");
        Instances dataSet = Util.readARFF("weather.nominal.arff");
        System.out.println(dataSet.toString());
        System.out.println("Class Attribute: " + dataSet.attribute(dataSet.classIndex()));

        System.out.println("\n========== Resampling Data Set ==========");
        dataSet = Util.resampleDataSet(dataSet);
        System.out.println(dataSet.toString());

        System.out.println("\n========== Reading File From CSV ==========");
        dataSet = Util.readCSV("weather.nominal.csv");
        System.out.println(dataSet.toString());
        System.out.println("Class Attribute: " + dataSet.attribute(dataSet.classIndex()));

        System.out.println("\n========== Removing Class Attributes ==========");
        dataSet = readARFF("weather.nominal.arff");
        dataSet = Util.removeAttribute(dataSet,dataSet.numAttributes());
        System.out.println(dataSet.toString());

        System.out.println("\n========== Building Naive Bayes Classifier ==========");
        dataSet = Util.readARFF("weather.nominal.arff");
        Classifier classifier = Util.buildClassifier(dataSet, new NaiveBayes());
        System.out.println(classifier.toString());

        System.out.println("\n========== Building ID3 Classifier ==========");
        dataSet = Util.readARFF("weather.nominal.arff");
        classifier = Util.buildClassifier(dataSet, new Id3());
        System.out.println(classifier.toString());

        System.out.println("\n========== Building J48 Classifier ==========");
        dataSet = Util.readARFF("weather.nominal.arff");
        classifier = Util.buildClassifier(dataSet, new J48());
        System.out.println(classifier.toString());

        System.out.println("\n========== Testing Naive Bayes Classifier ==========");
        dataSet = Util.readARFF("weather.nominal.arff");
        Instances trainSet = readARFF("weather.nominal.test.arff");
        classifier = Util.buildClassifier(dataSet, new NaiveBayes());
        Evaluation eval = Util.testClassifier(classifier, dataSet, trainSet);
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Testing ID3 Classifier ==========");
        classifier = Util.buildClassifier(dataSet, new Id3());
        eval = Util.testClassifier(classifier, dataSet, trainSet);
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Testing J48 Classifier ==========");
        classifier = Util.buildClassifier(dataSet, new J48());
        eval = Util.testClassifier(classifier, dataSet, trainSet);
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Cross Validation Naive Bayes Classifier ==========");
        eval = Util.crossValidationTest(dataSet, new NaiveBayes());
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Cross Validation ID3 Classifier ==========");
        eval = Util.crossValidationTest(dataSet, new Id3());
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Cross Validation J48 Classifier ==========");
        eval = Util.crossValidationTest(dataSet, new J48());
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Percentage Split Naive Bayes Classifier 80% ==========");
        eval = Util.percentageSplit(dataSet, new NaiveBayes(), 80);
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Percentage Split ID3 Classifier 80% ==========");
        eval = Util.percentageSplit(dataSet, new Id3(), 80);
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Percentage Split J48 Classifier 80% ==========");
        eval = Util.percentageSplit(dataSet, new J48(), 80);
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));

        System.out.println("\n========== Testing Save Model ==========");
        classifier = Util.buildClassifier(dataSet, new Id3());
        Util.saveModel("id3_weather_nominal.model", classifier);

        System.out.println("\n========== Testing Load Model ==========");
        System.out.println(Util.loadModel("id3_weather_nominal.model").toString());

        System.out.println("\n========== Classifying Model ==========");
        Util.classify("weather.nominal.classify.arff", classifier);
    }
}