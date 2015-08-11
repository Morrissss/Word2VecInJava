package main;

import training.VectorModel;
import training.Word2Vec;
import training.WordScore;

import java.util.List;

public class TestWord2Vec {

    public static void readByJava(String textFilePath, String tmpFilePath, String modelFilePath) throws Exception {

        Word2Vec wv = new Word2Vec.Factory()
		                .setMethod(Word2Vec.Method.SKIP_GRAM)
		                .setFreqThresold(5)
		                .setNumOfThread(10)
		                .setWindow(8)
		                .setSample(1e-4)
		                .setVectorSize(200).build();

        wv.training(textFilePath, tmpFilePath);
        wv.saveModel(modelFilePath);
    }

    public static void testVector(String modelFilePath) throws Exception {
        VectorModel vm = VectorModel.loadFromFile(modelFilePath);
        List<WordScore> result = vm.nearestTopN("亲", 20);
        for (WordScore we : result){
            System.out.println(we.name + " :\t" + we.score);
        }
    }

    public static void main(String[] args) throws Exception {
    	readByJava("/home/morris/github/Word2vecInJava/text8", 
    			   "/home/morris/github/Word2vecInJava/vocab", 
    			   "/home/morris/github/Word2vecInJava/vector_skip");
    }
}
