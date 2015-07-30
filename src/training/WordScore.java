package training;

public class WordScore implements Comparable<WordScore> {

    public String name;
    public double score;

    public WordScore(String name, double score) {
        this.name = name;
        this.score = score;
    }

    @Override
    public String toString() {
        return name + ":" + score;
    }

    @Override
    public int compareTo(WordScore o) {
    	return this.score < o.score ? 1 : (this.score == o.score ? 0 : -1);
    }
}
