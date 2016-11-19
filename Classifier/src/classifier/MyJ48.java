package classifier;

import J48.ErrorCalculator;
import J48.J48ClassDistribution;
import J48.NodeType;
import J48.Splitable;
import J48.Util;
import J48.NotSplitable;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.*;

import java.util.Enumeration;

public class MyJ48 extends Classifier {

    private MyJ48 [] childs;
    private boolean is_leaf;
    private boolean is_empty;
    private Instances dataSet;
    private double minimalInstances = 2;
    private J48ClassDistribution testSetDistribution;
    private float confidenceLevel = 0.1f;
    private NodeType nodeType;
    Instances [] subDataset;
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // Check if the data set is able to be proccessed using MyJ48.MyJ48
        getCapabilities().testWithFail(instances);

        Instances data = new Instances(instances);
        data.deleteWithMissingClass();

        createTree(data);

        collapseTree();
        pruneTree();
    }

    private void pruneTree() {
        int largestBranchIndex;
        double largestBranchError;
        double leafError;
        double treeError;
        MyJ48 largestBranch;

        if(!is_leaf)
        {
            for(int i=0; i<childs.length; i++)
            {
                childs[i].pruneTree();
            }

            largestBranchIndex = Utils.maxIndex(nodeType.classDistribution.weightPerSubDataset);
            largestBranchError = childs[largestBranchIndex].getBranchError(dataSet);
            leafError = getDistributionError(nodeType.classDistribution);
            treeError = getEstimatedTreeError();

            if(Utils.smOrEq(leafError, treeError+0.1) && Utils.smOrEq(leafError, largestBranchError+0.1))
            {
                childs = null;
                is_leaf = true;
                nodeType = new NotSplitable(nodeType.classDistribution);
            }
            else
            {
                if(Utils.smOrEq(largestBranchError, treeError + 0.1))
                {
                    largestBranch = childs[largestBranchIndex];
                    childs = largestBranch.childs;
                    nodeType = largestBranch.nodeType;
                    is_leaf = largestBranch.is_leaf;
                    createNewDistribution(dataSet);
                    pruneTree();
                }
            }
        }
    }

    private void createNewDistribution(Instances dataSet) {
        Instances [] subDataset;
        this.dataSet = dataSet;
        nodeType.resetDistribution(dataSet);
        if(!is_leaf)
        {
            subDataset = nodeType.split(dataSet);
            for(int i=0; i<childs.length; i++)
            {
                childs[i].createNewDistribution(subDataset[i]);
            }
        }
        else
        {
            if(!Utils.eq(0, dataSet.sumOfWeights()))
            {
                is_empty = false;
            }
            else
            {
                is_empty = true;
            }
        }
    }

    private double getEstimatedTreeError() {
        double error = 0;

        if(is_leaf)
        {
            return getDistributionError(nodeType.classDistribution);
        }
        else
        {
            for (int i=0; i<childs.length; i++)
            {
                error = error + childs[i].getEstimatedTreeError();
            }
            return error;
        }
    }

    private double getBranchError(Instances dataSet) {
        Instances [] subDataset;
        double error = 0;

        if(is_leaf)
        {
            return getDistributionError(new J48ClassDistribution(dataSet));
        }
        else
        {
            J48ClassDistribution tempClassDistribution = nodeType.classDistribution;
            nodeType.resetDistribution(dataSet);
            subDataset = nodeType.split(dataSet);
            nodeType.classDistribution = tempClassDistribution;
            for(int i=0; i<childs.length; i++)
            {
                error = error + childs[i].getBranchError(subDataset[i]);
                return error;
            }
        }
        return 0;
    }

    private double getDistributionError(J48ClassDistribution classDistribution) {
        if(Utils.eq(0, classDistribution.getTotalWeight())) {
            return 0;
        }
        else
        {
            return classDistribution.numIncorrect() + ErrorCalculator.calculateError(classDistribution.getTotalWeight(), classDistribution.numIncorrect(), confidenceLevel);
        }
    }

    public void collapseTree() {
        double subtreeError;
        double treeError;

        if (!is_leaf) {
            subtreeError = getTrainingError();
            treeError = nodeType.classDistribution.numIncorrect();
            if(subtreeError >= treeError-0.25)
            {
                childs = null;
                is_leaf = true;
                nodeType = new NotSplitable(nodeType.classDistribution);
            }
        }
        else
        {
            for (int i=0; i<childs.length; i++)
            {
                childs[i].collapseTree();
            }
        }
    }

    private void createTree(Instances data)
    {
        dataSet = data;
        is_leaf = false;
        is_empty = false;
        testSetDistribution = null;

        nodeType = processNode();
        if(nodeType.numOfSubsets > 1)
        {
            subDataset = nodeType.split(dataSet);
            childs = new MyJ48[nodeType.numOfSubsets];
            for(int i=0; i<nodeType.numOfSubsets; i++)
            {
                childs[i] = createNewTree(subDataset[i]);
            }
        }
        else
        {
            is_leaf = true;
            if(Utils.eq(dataSet.sumOfWeights(), 0))
            {
                is_empty = true;
            }
        }
    }

    private MyJ48 createNewTree(Instances subDataset) {
        MyJ48 newMyJ48 = new MyJ48();
        newMyJ48.createTree(subDataset);
        return newMyJ48;
    }

    private NodeType processNode()
    {
        double minGainRatio;
        Splitable[] splitables;
        Splitable bestSplitable = null;
        NotSplitable notSplitable = null;
        double averageInfoGain = 0;
        int usefulSplitables = 0;
        J48ClassDistribution classDistribution;
        double totalWeight;

        try{
            classDistribution = new J48ClassDistribution(dataSet);
            notSplitable = new NotSplitable(classDistribution);

            if(Utils.sm(dataSet.numInstances(), 2 * minimalInstances) ||
               Utils.eq(classDistribution.weightTotal, classDistribution.weightPerClass[Utils.maxIndex(classDistribution.weightPerClass)]))
            {
                return notSplitable;
            }

            splitables = new Splitable[dataSet.numAttributes()];

            Enumeration attributeEnumeration = dataSet.enumerateAttributes();
            while(attributeEnumeration.hasMoreElements())
            {
                Attribute attribute = (Attribute) attributeEnumeration.nextElement();
                splitables[attribute.index()] = new Splitable(attribute, minimalInstances, dataSet.sumOfWeights());
                splitables[attribute.index()].buildClassifier(dataSet);
                if(splitables[attribute.index()].validateNode())
                {
                    if(dataSet != null)
                    {
                        averageInfoGain = averageInfoGain +  splitables[attribute.index()].infoGain;
                        usefulSplitables++;
                    }
                }
            }

            if (usefulSplitables == 0)
            {
                return notSplitable;
            }
            averageInfoGain = averageInfoGain/(double)usefulSplitables;

            minGainRatio = 0;
            attributeEnumeration = dataSet.enumerateAttributes();
            while(attributeEnumeration.hasMoreElements())
            {
                Attribute attribute = (Attribute) attributeEnumeration.nextElement();
                if(splitables[attribute.index()].validateNode())
                {
                    if(splitables[attribute.index()].infoGain >= (averageInfoGain - 0.001) &&
                       Utils.gr(splitables[attribute.index()].gainRatio, minGainRatio))
                    {
                        bestSplitable = splitables[attribute.index()];
                        minGainRatio = bestSplitable.gainRatio;
                    }
                }
            }

            if (Utils.eq(minGainRatio,0))
            {
                return notSplitable;
            }

            bestSplitable.addInstanceWithMissingvalue();

            if(dataSet != null)
            {
                bestSplitable.setSplitPoint();
            }

            return bestSplitable;
        }

        catch (Exception e) {
            e.printStackTrace();
        };

        return null;
    }

    @Override
    public double classifyInstance(Instance instance)
            throws Exception {

        double maxProbability = Double.MAX_VALUE * -1;
        double currentProb;
        int maxIndex = 0;
        int j;

        for (j = 0; j < instance.numClasses(); j++) {
            currentProb = getProbs(j, instance);
            if (Utils.gr(currentProb,maxProbability)) {
                maxIndex = j;
                maxProbability = currentProb;
            }
        }

        return (double)maxIndex;
    }

    private double getProbs(int classIndex, Instance instance, double weight) {
        double prob = 0;

        if(is_leaf)
        {
            return weight * nodeType.classProb(classIndex, instance, -1);
        }
        else
        {
            int subsetIndex = nodeType.getSubsetIndex(instance);
            if(subsetIndex == -1)
            {
                double[] weights = nodeType.getWeights(instance);
                for(int i=0; i<childs.length; i++)
                {
                    if(!childs[i].is_empty)
                    {
                        prob += childs[i].getProbs(classIndex, instance, weights[i]*weight);
                    }
                }
                return prob;
            }
            else
            {
                if(childs[subsetIndex].is_empty)
                {
                    return weight * nodeType.classProb(classIndex, instance, subsetIndex);
                }
                else
                {
                    return childs[subsetIndex].getProbs(classIndex,instance,weight);
                }
            }
        }
    }

    private double getProbs(int classIndex, Instance instance) {
        return getProbs(classIndex, instance, 1);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return super.distributionForInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        /* Allowed attributes in MyJ48.MyJ48 */
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // Allowed class in MyJ48.MyJ48
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // Minimal instances for MyJ48.MyJ48
        result.setMinimumNumberInstances(0);

        return result;
    }

    public String toString() {

        try {
            StringBuffer text = new StringBuffer();

            if (is_leaf) {
                text.append(": ");
                text.append(nodeType.printLabel(0, dataSet));
            }else
                printTree(0, text);
            text.append("\n\nNumber of Leaves  : \t"+(numLeaves())+"\n");
            text.append("\nSize of the tree : \t"+numNodes()+"\n");

            return text.toString();
        } catch (Exception e) {
            return "Can't print classification tree.";
        }
    }

    public int numLeaves() {

        int num = 0;
        int i;

        if (is_leaf)
            return 1;
        else
            for (i=0;i<childs.length;i++)
                num = num+childs[i].numLeaves();

        return num;
    }

    public int numNodes() {

        int no = 1;
        int i;

        if (!is_leaf)
            for (i=0;i<childs.length;i++)
                no = no+childs[i].numNodes();

        return no;
    }

    private void printTree(int depth, StringBuffer text)
            throws Exception {

        int i,j;

        for (i=0;i<childs.length;i++) {
            text.append("\n");;
            for (j=0;j<depth;j++)
                text.append("|   ");
            text.append(nodeType.leftSide(dataSet));
            text.append(nodeType.rightSide(i, dataSet));
            if (childs[i].is_leaf) {
                text.append(": ");
                text.append(nodeType.printLabel(i, dataSet));
            }else
                childs[i].printTree(depth + 1, text);
        }
    }

    public double getTrainingError() {
        if(is_leaf)
        {
            return nodeType.classDistribution.numIncorrect();
        }
        else
        {
            double error = 0;
            for(int i=0; i<childs.length; i++)
            {
                error += childs[i].getTrainingError();
            }
            return error;
        }
    }

    public static void main (String [] args) throws Exception {
        
    }
}
