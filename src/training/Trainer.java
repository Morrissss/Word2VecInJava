package training;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Scanner;
import java.util.StringTokenizer;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import utils.MathUtils;
import huffman.HuffmanNode;
import huffman.HuffmanTree;
import huffman.WordNode;

public class Trainer {

    public enum Method {
        CBow, SKIP_GRAM
    }
    private Method trainMethod; // 神经网络学习方法
    private int windowSize; //文字窗口大小
    private int vectorSize; //词向量的元素个数
    private double subsampleRate;
//    private int negativeSample;
    private double initialAlpha;  // 初始学习率
    private int freqThreshold = 5;

    private Map<String, WordNode> wordNodeMap;
    private int corpusLen;     // 语料中的总词数
    private int threadNum;        // 线程个数
    private int iter;
    private int maxSentenceLen;

    public static class Factory {
        private int vectorSize = 200;
        private int windowSize = 5;
        private int freqThreshold = 3;
        private Method trainMethod = Method.SKIP_GRAM;
        private double sample = 1e-4;
//        private int negativeSample = 0;
        private double alpha = 0.05;
        private int numOfThread = 1;
        private int iter = 15;
        private int maxSentenceLen = 1000;

        public Factory setVectorSize(int size){
            vectorSize = size;
            return this;
        }
        public Factory setWindow(int size){
            windowSize = size;
            return this;
        }
        public Factory setFreqThreshold(int threshold){
            freqThreshold = threshold;
            return this;
        }
        public Factory setMethod(Method method){
            trainMethod = method;
            return this;
        }
        public Factory setSample(double rate){
            sample = rate;
            return this;
        }
//        public Factory setNegativeSample(int sample){
//            negativeSample = sample;
//            return this;
//        }
        public Factory setAlpha(double alpha){
            this.alpha = alpha;
            return this;
        }
        public Factory setNumOfThread(int numOfThread) {
            this.numOfThread = numOfThread;
            return this;
        }
        public Factory setIter(int iter) {
        	this.iter = iter;
        	return this;
        }
        public Factory setMaxSentenceLen(int maxSentenceLen) {
        	this.maxSentenceLen = maxSentenceLen;
        	return this;
        }
        
        public Trainer build(){
            return new Trainer(this);
        }
    }

    private Trainer(Factory factory) {
        vectorSize = factory.vectorSize;
        windowSize = factory.windowSize;
        freqThreshold = factory.freqThreshold;
        trainMethod = factory.trainMethod;
        subsampleRate = factory.sample;
//        negativeSample = factory.negativeSample;
        initialAlpha = factory.alpha;
        threadNum = factory.numOfThread;
        iter = factory.iter;
        corpusLen = 0;
        maxSentenceLen = factory.maxSentenceLen;
    }

    private void buildVocabulary(String inputFile, String outputFile) throws Exception {
        wordNodeMap = new HashMap<>(1<<16);
        try (BufferedReader br = new BufferedReader(new FileReader(inputFile))) {
            String line;
            corpusLen = 0;
            while ((line = br.readLine()) != null) {
            	StringTokenizer st = new StringTokenizer(line);
            	while (st.hasMoreTokens()) {
                	corpusLen++;
                    String word = st.nextToken();
                    if (wordNodeMap.containsKey(word)) {
                        WordNode node = wordNodeMap.get(word);
                        node.setFrequency(node.getFrequency() + 1);
                    } else {
                        wordNodeMap.put(word, new WordNode(word, 1, vectorSize));
                    }
                }
            }
        }
        System.out.println("Corpus size: " + corpusLen);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile))) {
	        Iterator<Map.Entry<String, WordNode>> iterator = wordNodeMap.entrySet().iterator();
	        while (iterator.hasNext()) {
	        	Map.Entry<String, WordNode> node = iterator.next();
	            if (node.getValue().getFrequency() < freqThreshold) {
	                iterator.remove();
	            } else {
	            	bw.write(node.getKey() + "\t" + node.getValue().getFrequency() + "\n");
	            }
	        }
	        System.out.println("Vocabulary size: " + wordNodeMap.size());
        }
        wordNodeMap = Collections.unmodifiableMap(wordNodeMap);
    }

    public void training(String fileName, String tmpFileName) throws Exception {
        buildVocabulary(fileName, tmpFileName);
        HuffmanNode root = HuffmanTree.makeHuffmanTree(wordNodeMap.values());

        ThreadTrainer.initParams(wordNodeMap, vectorSize, corpusLen, windowSize, initialAlpha, 
        		            	 trainMethod, root, iter);
        ExecutorService executor = Executors.newFixedThreadPool(threadNum);
        BlockingQueue<List<String>> corpusQueue = new ArrayBlockingQueue<>(1000);
        CountDownLatch latch = new CountDownLatch(threadNum);
        for (int i = 0; i < threadNum; i++) {
        	executor.execute(new ThreadTrainer(corpusQueue, latch));
        }
    	long totalCount = 0;
        for (int i = 0; i < iter; i++) {
	        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
	            String line;
	            while ((line = br.readLine()) != null) {
	                List<String> sentence = new ArrayList<>(maxSentenceLen+1);
	            	StringTokenizer st = new StringTokenizer(line);
	                while (st.hasMoreTokens()) {
	                    String word = st.nextToken();
	                    totalCount++;
	                    // sentence中的单词都在wordNodeMap中
	                    if (wordNodeMap.containsKey(word) && subsampleRate > 0) {
	                        if (include(wordNodeMap.get(word).getFrequency())) {
	                            sentence.add(word);
	                            if (sentence.size() >= maxSentenceLen) {
	                            	sentence.add(String.valueOf(totalCount));
	                            	corpusQueue.put(sentence);
	                            	sentence = new ArrayList<>(maxSentenceLen+1);
	                            }
	                        }
	                    }
	                }
	                if (!sentence.isEmpty()) {
                    	sentence.add(String.valueOf(totalCount));
	                	corpusQueue.put(sentence);
	                }
	            }
	        } catch (IOException e) {
	        	e.printStackTrace();
	        }
        }
        corpusQueue.put(ThreadTrainer.END_SENTENCE);
        latch.await();
        executor.shutdown();
        System.out.println("\nFinish");
    }

    // The subsampling randomly discards frequent words while keeping the ranking same
    private boolean include(int frequency) {
        double freqRatio = frequency / (subsampleRate * corpusLen);
        double rand = (Math.sqrt(freqRatio) + 1) / freqRatio;	// monotonously increasing with ratio
        return rand >= MathUtils.randomOne();
    }

    public void saveModel(String fileName) {
    	try (BufferedWriter bw = new BufferedWriter(new FileWriter(fileName))) {
    		bw.write(wordNodeMap.size() + " " + vectorSize + "\n");
    		for (Map.Entry<String, WordNode> entry : wordNodeMap.entrySet()) {
    			StringBuilder sb = new StringBuilder();
    			sb.append(entry.getKey());
    			for (float f : entry.getValue().getVector()) {
    				sb.append(" ").append(f);
    			}
    			sb.append("\n");
    			bw.write(sb.toString());
    		}
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private static List<WordNode> getNearest(Map<String, WordNode> wordNodeMap, String word, int num) {
    	final float[] vec = wordNodeMap.get(word).getVector();
    	PriorityQueue<WordNode> heap = new PriorityQueue<>(num, (x, y) -> compareProduct(x, y, vec));
        wordNodeMap.forEach((key, value) -> {
            if (!value.getWord().equals(word)) {
                if (heap.isEmpty()) {
                    heap.add(value);
                } else if (compareProduct(heap.peek(), value, vec) < 0) {
                    if (heap.size() == num) {
                        heap.poll();
                    }
                    heap.add(value);
                }
            }
        });
    	List<WordNode> result = new ArrayList<>(heap);
    	Collections.sort(result, (x, y) -> compareProduct(y, x, vec));
    	for (WordNode node : result) {
    		System.out.println(node.getWord() + ":\t" + MathUtils.cosineDis(node.getVector(), vec));
    	}
    	return result;
    }
    
    private static int compareProduct(WordNode a, WordNode b, float[] vec) {
    	double p1 = MathUtils.cosineDis(a.getVector(), vec);
    	double p2 = MathUtils.cosineDis(b.getVector(), vec);
    	return Double.compare(p1, p2);
    }
    
    public static void main(String[] args) throws IOException {
    	try (BufferedReader br = new BufferedReader(new FileReader(
    			"/home/morris/github/Word2vecInJava/vector"))) {
    		String line = "";
    		String[] parts = br.readLine().split(" ");
    		int wordNum = Integer.parseInt(parts[0]);
    		int vectorSize = Integer.parseInt(parts[1]);
    		Map<String, WordNode> wordNodeMap = new HashMap<>(wordNum);
    		while ((line = br.readLine()) != null) {
    			parts = line.split(" ");
    			String word = parts[0];
    			WordNode node = new WordNode(word, 0, vectorSize);
    			float[] vec = new float[vectorSize];
    			for (int i = 0; i < vec.length; i++)
    				vec[i] = Float.parseFloat(parts[i+1]);
    			node.setVector(vec);
    			wordNodeMap.put(word, node);
    		}
    		
            Scanner sc = new Scanner(System.in);
            while (sc.hasNext()) {
            	String word = sc.next();
            	getNearest(wordNodeMap, word, 10);
            }
            sc.close();
    	}
    }
}