package huffman;

import java.util.Arrays;

import utils.MathUtils;

public class WordNode extends HuffmanNode {

    private String word;

    public String getWord() {
        return word;
    }

	public void setWord(String word) {
        this.word = word;
    }

    public WordNode(String word, int freq, int vectorSize, boolean useNeg) {
        super(freq, vectorSize, useNeg);
        this.word = word;
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) (MathUtils.randomOne() - 0.5) / vectorSize;
        }
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("WordNode{");
        sb.append("word='").append(word).append('\'');
        sb.append(", frequency=").append(frequency);
        sb.append(", code=").append(code);
        sb.append(", parent=").append(parent);
        sb.append(", vector=").append(Arrays.toString(vector));
        sb.append(", negVector=").append(Arrays.toString(negVector));
        sb.append('}');
        return sb.toString();
    }
}
