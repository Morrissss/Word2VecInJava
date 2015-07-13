package huffman;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class HuffmanNode implements Comparable<HuffmanNode> {

    protected int frequency = 0;
    protected int code = 0;
    protected HuffmanNode parent;
    protected float[] vector;
    
    /**
     * 包含本节点，不包含根节点
     */
    public List<HuffmanNode> findPath(HuffmanNode root) {
    	List<HuffmanNode> result = new ArrayList<>();
    	for (HuffmanNode p = this; p != root; p = p.parent)
    		result.add(p);
    	Collections.reverse(result);
    	return result;
    }

    public int getFrequency() {
        return frequency;
    }
    public void setFrequency(int frequency) {
        this.frequency = frequency;
    }

    public int getCode() {
        return code;
    }
    public void setCode(int code) {
        this.code = code;
    }

    public HuffmanNode getParent() {
        return parent;
    }
    public void setParent(HuffmanNode parent) {
        this.parent = parent;
    }

    public float[] getVector() {
        return vector;
    }
    public void setVector(float[] vector) {
        this.vector = vector;
    }

	public static HuffmanNode merge(HuffmanNode left, HuffmanNode right) {
        HuffmanNode result = new HuffmanNode(left.frequency + right.frequency, left.vector.length);
        left.parent = right.parent = result;
        left.code = 0;
        right.code = 1;
        return result;
    }

    @Override
    public int compareTo(HuffmanNode hn) {
        return (this.frequency > hn.frequency) ? 1 :
                this.frequency == hn.frequency ? 0 : -1;
    }

    public HuffmanNode(int freq, int vectorSize) {
        this.frequency = freq;
        vector = new float[vectorSize];
        parent = null;
        code = -1;
    }

    @Override
	public String toString() {
		return "HuffmanNode [frequency=" + frequency + ", code=" + code
				+ ", parent=" + parent + ", vector=" + Arrays.toString(vector)
				+ "]";
	}
}
