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
    protected float[] negVector;
    
    /**
     * 包含本节点，不包含根节点
     */
    public List<HuffmanNode> findPath(HuffmanNode root) {
    	List<HuffmanNode> result = new ArrayList<>();
    	for (HuffmanNode p = this; p != root; p = p.parent) {
            result.add(p);
        }
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

    public float[] getNegVector() {
        return negVector;
    }
    public void setNegVector(float[] negVector) {
        this.negVector = negVector;
    }

    public static HuffmanNode merge(HuffmanNode left, HuffmanNode right, boolean useNeg) {
        HuffmanNode result = new HuffmanNode(left.frequency + right.frequency, left.vector.length, useNeg);
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

    public HuffmanNode(int freq, int vectorSize, boolean useNeg) {
        this.frequency = freq;
        vector = new float[vectorSize];
        parent = null;
        code = -1;
        if (useNeg) {
            vector = new float[vectorSize];
        }
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("HuffmanNode{");
        sb.append("frequency=").append(frequency);
        sb.append(", code=").append(code);
        sb.append(", parent=").append(parent);
        sb.append(", vector=").append(Arrays.toString(vector));
        sb.append(", negVector=").append(Arrays.toString(negVector));
        sb.append('}');
        return sb.toString();
    }
}
