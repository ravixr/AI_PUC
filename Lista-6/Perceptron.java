import java.util.Random;

public class Perceptron {
    private double[] weights;
    private double bias;
    private double learningRate;

    public Perceptron(int numInputs, double learningRate) {
        this.weights = new double[numInputs];
        this.bias = 0;
        this.learningRate = learningRate;
        Random random = new Random();
        for (int i = 0; i < numInputs; i++) {
            this.weights[i] = random.nextDouble();
        }
    }

    public int classify(double[] inputs) {
        double activation = 0;
        for (int i = 0; i < inputs.length; i++) {
            activation += inputs[i] * this.weights[i];
        }
        activation += this.bias;
        if (activation >= 0) {
            return 1;
        } else {
            return 0;
        }
    }

    public void train(double[][] inputs, int[] labels, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                int predicted = this.classify(inputs[i]);
                int error = labels[i] - predicted;
                for (int j = 0; j < inputs[i].length; j++) {
                    this.weights[j] += error * inputs[i][j] * this.learningRate;
                }
                this.bias += error * this.learningRate;
            }
        }
    }

    public static void main(String[] args) {
        // Criar perceptrons para as funções AND, OR e XOR
        Perceptron andPerceptron = new Perceptron(2, 0.1);
        Perceptron orPerceptron = new Perceptron(2, 0.1);
        Perceptron xorPerceptron = new Perceptron(2, 0.1);
    
        // Dados de treinamento para AND
        double[][] andInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] andLabels = {0, 0, 0, 1};
    
        // Treinar perceptron AND
        andPerceptron.train(andInputs, andLabels, 100);
    
        // Testar perceptron AND
        System.out.println("AND");
        System.out.println("0 AND 0 = " + andPerceptron.classify(new double[]{0, 0}));
        System.out.println("0 AND 1 = " + andPerceptron.classify(new double[]{0, 1}));
        System.out.println("1 AND 0 = " + andPerceptron.classify(new double[]{1, 0}));
        System.out.println("1 AND 1 = " + andPerceptron.classify(new double[]{1, 1}));
    
        // Dados de treinamento para OR
        double[][] orInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] orLabels = {0, 1, 1, 1};
    
        // Treinar perceptron OR
        orPerceptron.train(orInputs, orLabels, 100);
    
        // Testar perceptron OR
        System.out.println("OR");
        System.out.println("0 OR 0 = " + orPerceptron.classify(new double[]{0, 0}));
        System.out.println("0 OR 1 = " + orPerceptron.classify(new double[]{0, 1}));
        System.out.println("1 OR 0 = " + orPerceptron.classify(new double[]{1, 0}));
        System.out.println("1 OR 1 = " + orPerceptron.classify(new double[]{1, 1}));
    
        // Dados de treinamento para XOR
        double[][] xorInputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] xorLabels = {0, 1, 1, 0};
    
        // Treinar perceptron XOR
        xorPerceptron.train(xorInputs, xorLabels, 100);
    
        // Testar perceptron XOR
        System.out.println("XOR");
        System.out.println("0 XOR 0 = " + xorPerceptron.classify(new double[]{0, 0}));
        System.out.println("0 XOR 1 = " + xorPerceptron.classify(new double[]{0, 1}));
        System.out.println("1 XOR 0 = " + xorPerceptron.classify(new double[]{1, 0}));
        System.out.println("1 XOR 1 = " + xorPerceptron.classify(new double[]{1, 1}));
    }
    
}
