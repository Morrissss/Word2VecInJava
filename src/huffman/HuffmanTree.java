package huffman;

import java.util.*;

public class HuffmanTree {

    public static HuffmanNode makeHuffmanTree(Collection<? extends HuffmanNode> nodes) {
        PriorityQueue<HuffmanNode> heap = new PriorityQueue<>(nodes);
        HuffmanNode parent = null;
        while (heap.size() > 1){
            HuffmanNode left = heap.poll();
            HuffmanNode right = heap.poll();
            parent = HuffmanNode.merge(left, right);
            heap.add(parent);
        }
        System.out.println("Network initialized");
        return parent;
    }
}
