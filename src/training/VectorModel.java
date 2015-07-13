package training;

import utils.MathUtils;

import java.io.*;
import java.util.*;

public class VectorModel {

    private Map<String, float[]> wordToVectorMap = new HashMap<String, float[]>();
    private int wordVecDim = 200;
    private int topNSize = 40;

    public VectorModel(Map<String, float[]> wordToVectorMap, int wordVecDim) {
        if (wordToVectorMap == null || wordToVectorMap.isEmpty()) {
            throw new IllegalArgumentException("word2vec的词向量为空，请先训练模型。");
        }
        if (wordVecDim <= 0) {
            throw new IllegalArgumentException("词向量长度（layerSize）应大于0");
        }
        this.wordToVectorMap = wordToVectorMap;
        this.wordVecDim = wordVecDim;
    }


    /**
     * 使用Word2Vec保存的模型加载词向量模型
     * @param path 模型文件路径
     * @return 词向量模型
     */
    public static VectorModel loadFromFile(String path) throws Exception {
        if (path == null || path.isEmpty()){
            throw new IllegalArgumentException("模型路径可以为null或空。");
        }
        Map<String, float[]> wordMapLoaded = new HashMap<>(1<<16);
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line = br.readLine();
            String[] parts = line.split(" ");
            int wordCount = Integer.parseInt(parts[0]);
            int wordVectorDim = Integer.parseInt(parts[1]);

            for (int i = 0; i < wordCount; i++) {
                line = br.readLine();
                parts = line.split(" ");
                String word = parts[0];
                float[] value = new float[wordVectorDim];
                for (int j = 0; j < wordVectorDim; j++) {
                    value[j] = Float.parseFloat(parts[j+1]);
                }
                MathUtils.normalizeVector(value);
                wordMapLoaded.put(word, value);
            }
            return new VectorModel(wordMapLoaded, wordVectorDim);
        }
    }

    /**
     * 保存词向量模型
     * @param file 模型存放路径
     */
    public void saveModel(File file) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(file))) {
            bw.write(wordToVectorMap.size() + " " + wordVecDim + "\n");
            for (Map.Entry<String, float[]> element : wordToVectorMap.entrySet()) {
                StringBuilder sb = new StringBuilder();
                sb.append(element.getKey());
                for (float d : element.getValue()) {
                    sb.append(" " + d);
                }
            }
        }
    }

    /**
     * 获取与词word最相近topNSize个词
     * @param queryWord 词
     * @return 相近词集，若模型不包含词word，则返回空集
     */
    public Set<WordScore> similar(String queryWord){

        float[] center = wordToVectorMap.get(queryWord);
        if (center == null){
            return Collections.emptySet();
        }

        int resultSize = wordToVectorMap.size() < topNSize ? wordToVectorMap.size() : topNSize + 1;
        TreeSet<WordScore> result = new TreeSet<WordScore>();
        for (int i = 0; i < resultSize; i++){
            result.add(new WordScore("^_^", -Float.MAX_VALUE));
        }
        float minDist = -Float.MAX_VALUE;
        for (Map.Entry<String, float[]> entry : wordToVectorMap.entrySet()){
            float[] vector = entry.getValue();
            float dist = 0;
            for (int i = 0; i < vector.length; i++){
                dist += center[i] * vector[i];
            }
            if (dist > minDist){
                result.add(new WordScore(entry.getKey(), dist));
                minDist = result.pollLast().score;
            }
        }
        result.pollFirst();

        return result;
    }

    public Set<WordScore> similar(float[] center) {
        if (center == null || center.length != wordVecDim){
            return Collections.emptySet();
        }

        int resultSize = wordToVectorMap.size() < topNSize ? wordToVectorMap.size() : topNSize;
        TreeSet<WordScore> result = new TreeSet<WordScore>();
        for (int i = 0; i < resultSize; i++){
            result.add(new WordScore("^_^", -Float.MAX_VALUE));
        }
        float minDist = -Float.MAX_VALUE;
        for (Map.Entry<String, float[]> entry : wordToVectorMap.entrySet()){
            float[] vector = entry.getValue();
            float dist = 0;
            for (int i = 0; i < vector.length; i++){
                dist += center[i] * vector[i];
            }
            if (dist > minDist){
                result.add(new WordScore(entry.getKey(), dist));
                minDist = result.pollLast().score;
            }
        }
//        result.pollFirst();

        return result;
    }

    /**
     * 词迁移，即word1 - word0 + word2 的结果，若三个词中有一个不在模型中，
     * 也就是没有词向量，则返回空集
     * @param word0 词
     * @param word1 词
     * @param word2 词
     * @return 与结果最相近的前topNSize个词
     */
    public TreeSet<WordScore> analogy(String word0, String word1, String word2) {
        float[] wv0 = wordToVectorMap.get(word0);
        float[] wv1 = wordToVectorMap.get(word1);
        float[] wv2 = wordToVectorMap.get(word2);

        if (wv1 == null || wv2 == null || wv0 == null) {
            return null;
        }
        float[] center = new float[wordVecDim];
        for (int i = 0; i < wordVecDim; i++) {
            center[i] = wv1[i] - wv0[i] + wv2[i];
        }

        int resultSize = wordToVectorMap.size() < topNSize ? wordToVectorMap.size() : topNSize;
        TreeSet<WordScore> result = new TreeSet<WordScore>();
        for (int i = 0; i < resultSize; i++){
            result.add(new WordScore("^_^", -Float.MAX_VALUE));
        }
        String name;
        float minDist = -Float.MAX_VALUE;
        for (Map.Entry<String, float[]> entry : wordToVectorMap.entrySet()){
            name = entry.getKey();
            if (name.equals(word1) || name.equals((word2))){
                continue;
            }
            float[] vector = entry.getValue();
            float dist = 0;
            for (int i = 0; i < vector.length; i++){
                dist += center[i] * vector[i];
            }
            if (dist > minDist){
                result.add(new WordScore(entry.getKey(), dist));
                minDist = result.pollLast().score;
            }
        }
        return result;
    }

    public float[] getWordVector(String word) {
        return wordToVectorMap.get(word);
    }


    public class WordScore implements Comparable<WordScore> {

        public String name;
        public float score;

        public WordScore(String name, float score) {
            this.name = name;
            this.score = score;
        }

        @Override
        public String toString() {
            return this.name + "\t" + score;
        }

        @Override
        public int compareTo(WordScore o) {
            if (this.score < o.score) {
                return 1;
            } else {
                return -1;
            }
        }
    }

}
