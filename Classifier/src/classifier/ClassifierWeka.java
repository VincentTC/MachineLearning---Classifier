/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
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
    
    public static Instances loadData(String filename) throws Exception {
        DataSource source = new DataSource("data/"+ filename);
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        
        return data;
    }
    
    public static void removeAttribute(String index){
        Remove rm = new Remove();
        rm.setAttributeIndices(index);
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(rm);
    }
    
    public static Instances resample(Instances data){
        final Resample filter = new Resample();
	Instances filteredIns = null;
	filter.setBiasToUniformClass(1.0);
	try {
		filter.setInputFormat(data);
		filter.setNoReplacement(false);
		filter.setSampleSizePercent(100);
		filteredIns = Filter.useFilter(data, filter);
	} catch (Exception e) {
		System.out.println("Error when resampling input data!");
		e.printStackTrace();
	}
	return filteredIns;
    }
    
    public static void buildClassifier(Classifier classifierModel, Instances data) throws Exception {
        classifierModel.buildClassifier(data);
    }
    
    public static void crossValidation(Classifier classifierModel, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifierModel, data, 10, new Random(1));
        System.out.println("Accuracy: "+Double.toString(eval.pctCorrect()));
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
    }
    
}
