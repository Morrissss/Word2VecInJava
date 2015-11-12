package training;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import utils.MathUtils;

public class VectorModel {

    private Map<String, float[]> wordToVectorMap = new HashMap<String, float[]>();
    private int wordVecDim = 200;

    public VectorModel(Map<String, float[]> wordToVectorMap) {
        if (wordToVectorMap == null || wordToVectorMap.isEmpty()) {
            throw new IllegalArgumentException("wordToVectorMap should not be empty");
        }
        this.wordToVectorMap = wordToVectorMap;
        for (Map.Entry<String, float[]> wordVec : wordToVectorMap.entrySet()) {
            this.wordVecDim = wordVec.getValue().length;
            break;
        }
    }

    public static VectorModel loadFromFile(String path) throws Exception {
        if (path == null || path.isEmpty()) {
            throw new IllegalArgumentException("path should not be empty");
        }
        Map<String, float[]> wordMapLoaded = new HashMap<>(1 << 16);
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
                    value[j] = Float.parseFloat(parts[j + 1]);
                }
                MathUtils.normalizeVector(value);
                wordMapLoaded.put(word, value);
            }
            return new VectorModel(wordMapLoaded);
        }
    }

    public void saveToFile(String path) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
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

    public List<WordScore> nearestTopN(String queryWord, int topNSize) {
        float[] center = wordToVectorMap.get(queryWord);
        if (center == null) {
            return Collections.emptyList();
        }
        return findNearest(center, queryWord, topNSize);
    }

    /**
     * 词迁移，即word1 - word0 + word2 的结果，若三个词中有一个不在模型中，
     * 也就是没有词向量，则返回空集
     *
     * @return 与结果最相近的前topNSize个词
     */
    public List<WordScore> analogy(String wordFrom, String wordTo, String wordTarget,
                                   int topNSize) {
        float[] vecFrom = wordToVectorMap.get(wordFrom);
        float[] vecTo = wordToVectorMap.get(wordTo);
        float[] vecTarget = wordToVectorMap.get(wordTarget);
        if (vecTo == null || vecTarget == null || vecFrom == null) {
            return Collections.emptyList();
        }

        float[] targetCenter = MathUtils.vectorAdd(vecTarget, MathUtils.vectorMinus(vecTo, vecFrom));
        return findNearest(targetCenter, wordTarget, topNSize);
    }

    private List<WordScore> findNearest(float[] center, String excludeWord, int num) {
        PriorityQueue<WordScore> topHeap = new PriorityQueue<>(num);
        for (Map.Entry<String, float[]> entry : wordToVectorMap.entrySet()) {
            String word = entry.getKey();
            if (word.equals(excludeWord)) {
                continue;
            }
            double dist = MathUtils.dotProduct(center, entry.getValue());
            if (topHeap.isEmpty()) {
                topHeap.add(new WordScore(word, dist));
            } else if (topHeap.peek().score > dist) {
                if (topHeap.size() == num) {
                    topHeap.poll();
                }
                topHeap.add(new WordScore(word, dist));
            }
        }
        List<WordScore> result = new ArrayList<>(topHeap);
        Collections.sort(result, Collections.reverseOrder());
        return result;
    }
}
