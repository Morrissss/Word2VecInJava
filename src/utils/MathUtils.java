package utils;

import java.util.concurrent.atomic.AtomicLong;

public class MathUtils {

    private static final int EXP_TABLE_SIZE = 1000;
    private static final int MAX_EXP = 6;
    private static final double[] EXP_TABLE;
    static {
        EXP_TABLE = new double[EXP_TABLE_SIZE];
        for (int i = 0; i < EXP_TABLE_SIZE; i++) {
            EXP_TABLE[i] = Math.exp(((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
            EXP_TABLE[i] = EXP_TABLE[i] / (EXP_TABLE[i] + 1);
        }
    }

    private static AtomicLong rand = new AtomicLong(5);


    public static double sigmoid(double x) {
        if (x <= -MAX_EXP)
            return 0;
        if (x >= MAX_EXP)
            return 1;
        return EXP_TABLE[(int) ((x + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    }

    public static long random() {	
        return rand.updateAndGet(x -> x * 25214903917L + 11);
    }

    public static double randomOne() {
        long r = random();
        return (r & 0xFFFF) / (double) 65536;
    }

    public static int random(int lowerBound, int upperBound) {
    	int mod = upperBound-lowerBound+1;
        return lowerBound + (int) (random() % mod + mod) % mod;
    }
    
    public static double norm2(float[] vec) {
        if (vec == null || vec.length == 0)
            return Double.NaN;
        double norm2 = 0.0;
        for (double e : vec) {
            norm2 += e * e;
        }
        return Math.sqrt(norm2);
    }

    public static void normalizeVector(float[] vec) {
        double norm = norm2(vec);
        if (norm > 0) {
            for (int i = 0; i < vec.length; i++)
                vec[i] /= norm;
        }
    }

    public static float[] vectorAdd(float[] vec1, float[] vec2) {
        if (vec1 == null || vec2 == null || vec1.length != vec2.length)
            return null;
        float[] result = new float[vec1.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = vec1[i] + vec2[i];
        }
        return result;
    }

    public static float[] vectorMinus(float[] vec1, float[] vec2) {
        if (vec1 == null || vec2 == null || vec1.length != vec2.length)
            return null;
        float[] result = new float[vec1.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = vec1[i] - vec2[i];
        }
        return result;
    }

    public static void vectorAcc(float[] vec1, float[] vec2) {
        if (vec2 == null || vec2.length != vec1.length)
            return;
        for (int i = 0; i < vec1.length; i++) {
            vec1[i] += vec2[i];
        }
    }
    public static void vectorAcc(double[] vec1, double[] vec2) {
        if (vec2 == null || vec2.length != vec1.length)
            return;
        for (int i = 0; i < vec1.length; i++) {
            vec1[i] += vec2[i];
        }
    }

    public static void vectorSub(float[] vec1, float[] vec2) {
        if (vec2 == null || vec2.length != vec1.length)
            return;
        for (int i = 0; i < vec1.length; i++) {
            vec1[i] -= vec2[i];
        }
    }
    public static void vectorSub(double[] vec1, double[] vec2) {
        if (vec2 == null || vec2.length != vec1.length)
            return;
        for (int i = 0; i < vec1.length; i++) {
            vec1[i] -= vec2[i];
        }
    }

    public static float[] vectorScale(float[] vec, double scale) {
        if (vec == null)
            return null;
        float[] result = new float[vec.length];
        for (int i = 0; i < vec.length; i++) {
            result[i] = (float) (vec[i] * scale);
        }
        return result;
    }
    public static double[] vectorScale(double[] vec, double scale) {
        if (vec == null)
            return null;
        double[] result = new double[vec.length];
        for (int i = 0; i < vec.length; i++) {
            result[i] = (float) (vec[i] * scale);
        }
        return result;
    }

    public static double dotProduct(float[] vec1, float[] vec2) {
        if (vec1 == null || vec2 == null || vec1.length != vec2.length)
            return Double.NaN;
        double result = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            result += vec1[i] * vec2[i];
        }
        return result;
    }
    public static double dotProduct(double[] vec1, double[] vec2) {
        if (vec1 == null || vec2 == null || vec1.length != vec2.length)
            return Double.NaN;
        double result = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            result += vec1[i] * vec2[i];
        }
        return result;
    }
    
    public static double cosineDis(float[] vec1, float[] vec2) {
    	return dotProduct(vec1, vec2) / (norm2(vec1) * norm2(vec2));
    }
}