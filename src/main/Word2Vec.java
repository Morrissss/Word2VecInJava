package main;

import training.Trainer;
import training.VectorModel;

import java.util.Collections;
import java.util.Set;

public class Word2Vec {

    public static void readByJava(String textFilePath, String tmpFilePath, String modelFilePath) throws Exception {

        Trainer wv = new Trainer.Factory()
                .setMethod(Trainer.Method.CBow)
                .setFreqThreshold(5)
                .setNumOfThread(10)
                .setWindow(8)
                .setSample(1e-4)
                .setVectorSize(200).build();

        wv.training(textFilePath, tmpFilePath);
        wv.saveModel(modelFilePath);
    }

    public static void testVector(String modelFilePath) throws Exception {
        VectorModel vm = VectorModel.loadFromFile(modelFilePath);
        Set<VectorModel.WordScore> result1 = Collections.emptySet();

        result1 = vm.nearestTopN("亲");
        for (VectorModel.WordScore we : result1){
            System.out.println(we.name + " :\t" + we.score);
        }
    }

    public static void main(String[] args) throws Exception {
    	readByJava("/home/morris/github/Word2vecInJava/test",
    			   "/home/morris/github/Word2vecInJava/vocab", 
    			   "/home/morris/github/Word2vecInJava/vector");
    }
}
