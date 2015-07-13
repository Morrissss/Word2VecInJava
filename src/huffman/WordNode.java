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

    public WordNode(String word, int freq, int vectorSize) {
        super(freq, vectorSize);
        this.word = word;
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (float) (MathUtils.randomOne() - 0.5) / vectorSize;
        }
    }

	@Override
	public String toString() {
		return "WordNode [word=" + word + ", frequency=" + frequency
				+ ", code=" + code + ", parent=" + parent + ", vector="
				+ Arrays.toString(vector) + "]";
	}
}
