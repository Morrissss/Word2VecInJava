package training;

import training.Trainer.Method;
import huffman.HuffmanNode;
import huffman.WordNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static utils.MathUtils.*;

public class ThreadTrainer implements Runnable {
	
	public static final List<String> END_SENTENCE = Collections.unmodifiableList(new ArrayList<>());

	private static Map<String, WordNode> wordNodeMap;
    private static int vectorDim;
    private static int corpusLen;
    private static double initialAlpha;
    private static int windowSize;
    private static Method trainMethod;
    private static HuffmanNode root;
    private static int iter;
    
    public static void initParams(Map<String, WordNode> wordNodeMap, int vectorDim, 
    		                      int corpusLen, int windowSize, double initAlpha, 
    							  Method trainMethod, HuffmanNode root, int iter) {
    	ThreadTrainer.wordNodeMap = wordNodeMap;
    	ThreadTrainer.vectorDim = vectorDim;
    	ThreadTrainer.corpusLen = corpusLen;
    	ThreadTrainer.windowSize = windowSize;
    	ThreadTrainer.initialAlpha = initAlpha;
    	ThreadTrainer.trainMethod = trainMethod;
    	ThreadTrainer.root = root;
    	ThreadTrainer.iter = iter;
    }

    private BlockingQueue<List<String>> corpusQueue;
    private CountDownLatch latch;

    public ThreadTrainer(BlockingQueue<List<String>> corpusQueue, CountDownLatch latch) {
        this.corpusQueue = corpusQueue;
        this.latch = latch;
    }

    private double computeAlpha(long totalTrained) {
        double alpha = Math.max(initialAlpha * (1 - totalTrained / (double) (iter*corpusLen + 1)), 
        						initialAlpha * 0.0001);	// threshold
        System.out.print(String.format("\rAlpha: %f  Process: %.2f%%",
                		 alpha, 100 * totalTrained / (double) (iter * corpusLen + 1)));
        System.out.flush();
        return alpha;
    }

    private void training(List<String> sentence, double alpha) {
        for (int index = 0; index < sentence.size(); index++) {
            int extractedWordNum = random(0, windowSize-1);
            if (trainMethod == Trainer.Method.CBow) {
                cbowGram(index, sentence, extractedWordNum, alpha);
            } else {
                skipGram(index, sentence, extractedWordNum, alpha);
            }
        }
    }

    @Override
    public void run() {
        try {
            while (true) {
                List<String> sentence = corpusQueue.poll(10, TimeUnit.SECONDS);
                if (sentence != null) {
	                if (sentence != END_SENTENCE) {
	                    double alpha = computeAlpha(Long.parseLong(sentence.get(sentence.size()-1)));
	                    sentence.remove(sentence.size()-1);
	                    training(sentence, alpha);
	                } else {
	                	corpusQueue.add(END_SENTENCE);
	                	latch.countDown();
	                	break;
	                }
                }
            }
            System.out.println(Thread.currentThread() + ": " + latch.getCount());
        } catch (Exception e) {
        	e.printStackTrace(System.err);
        }
    }

    private void skipGram(int index, List<String> sentence, int extractedNum, double alpha) {
        WordNode word = wordNodeMap.get(sentence.get(index));
        
        int contextLowerBound = Math.max(0, index - windowSize + extractedNum);
        int contextUpperBound = Math.min(sentence.size()-1, index + windowSize - extractedNum);
        for (int i = contextLowerBound; i <= contextUpperBound; i++) {
        	if (i != index) {
	            float[] neu1e = new float[vectorDim];
	            Arrays.fill(neu1e, 0);
	            //Hierarchical Softmax
	            float[] contextWordVec = wordNodeMap.get(sentence.get(i)).getVector();
	            synchronized (contextWordVec) {
		            for (HuffmanNode pathNode : word.findPath(root)){
		                float[] parentVec = pathNode.getParent().getVector();
		                synchronized (parentVec) {
			                double f = dotProduct(contextWordVec, parentVec);
			                // Propagate hidden -> output
			                f = sigmoid(f);
			                if (f == 0 || f == 1) {
			                    continue;
			                }
			                // 'g' is the gradient multiplied by the learning rate
			                double g = (1 - pathNode.getCode() - f) * alpha;
		                	// Propagate errors output -> hidden
			                vectorAcc(neu1e, vectorScale(parentVec, g));
			                // Learn weights hidden -> output
			                vectorAcc(parentVec, vectorScale(contextWordVec, g));
		                }
		            }
		            // Learn weights input -> hidden
		            vectorAcc(contextWordVec, neu1e);
	            }
        	}
        }
    }

    private void cbowGram(int index, List<String> sentence, int extractedNum, double alpha) {
        WordNode word = wordNodeMap.get(sentence.get(index));

        int contextLowerBound = Math.max(0, index - windowSize + extractedNum);
        int contextUpperBound = Math.min(sentence.size()-1, index + windowSize - extractedNum);
        if (contextUpperBound - contextLowerBound <= 0)
        	return;
        float[] neu1 = new float[vectorDim];
        Arrays.fill(neu1, 0);
        for (int i = contextLowerBound; i <= contextUpperBound; i++) {
        	if (i != index) {
	        	float[] contextWordVec = wordNodeMap.get(sentence.get(i)).getVector();
            	synchronized (contextWordVec) {
            		vectorAcc(neu1, contextWordVec);
            	}
        	}
        }
        neu1 = vectorScale(neu1, 1 / (double) (contextUpperBound-contextLowerBound));
        //Hierarchical Softmax
        float[] neu1e = new float[vectorDim];
        Arrays.fill(neu1e, 0);
        
        for (HuffmanNode pathNode : word.findPath(root)) {
            // Propagate hidden -> output
        	double f = dotProduct(neu1, pathNode.getParent().getVector());
            f = sigmoid(f);
            if (f == 0 || f == 1) {
                continue;
            }
            // 'g' is the gradient multiplied by the learning rate
            double g = (1 - pathNode.getCode() - f) * alpha;
            float[] parentVec = pathNode.getParent().getVector();
            synchronized (parentVec) {
                // Propagate errors output -> hidden
	            vectorAcc(neu1e, vectorScale(parentVec, g));
	            // Learn weights hidden -> output
	            vectorAcc(parentVec, vectorScale(neu1, g));
            }
        }
        for (int i = contextLowerBound; i <= contextUpperBound; i++) {
        	if (i != index) {
	        	float[] contextWordVec = wordNodeMap.get(sentence.get(i)).getVector();
	        	synchronized (contextWordVec) {
	                vectorAcc(contextWordVec, neu1e);
	        	}
        	}
        }
    }
}
