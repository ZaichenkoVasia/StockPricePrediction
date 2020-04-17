package com.mycompany.stockprediction;

import com.opencsv.CSVReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Iterator for stock data set
 */
public class StockDataSetIterator implements DataSetIterator {

    private final int VECTOR_SIZE = 1;
    private int miniBatchSize;
    private int exampleLength;
    private double[] minNum = new double[VECTOR_SIZE];
    private double[] maxNum = new double[VECTOR_SIZE];

    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    private List<StockPrice> train;
    private List<Pair> test;

    public StockDataSetIterator(String filename, int miniBatchSize, int exampleLength, int firstTestItemNumber, int testItems) {
        List<StockPrice> stockPriceList = readStockDataFromFile(filename);
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;

        train = stockPriceList.subList(0, firstTestItemNumber);

        test = generateTestDataSet(stockPriceList.subList(firstTestItemNumber, firstTestItemNumber + testItems));
        initializeOffsets();
    }

    private void initializeOffsets() {
        exampleStartOffsets.clear();
        int window = exampleLength + 1;
        for (int i = 0; i < train.size() - window; i++) {
            exampleStartOffsets.add(i);
        }
    }

    public List<Pair> getTestDataSet() {
        return test;
    }

    public double[] getMaxNum() {
        return maxNum;
    }

    public double[] getMinNum() {
        return minNum;
    }

    @Override
    public DataSet next(int num) {
        if (exampleStartOffsets.size() == 0) throw new NoSuchElementException();
        int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());
        INDArray input = Nd4j.create(new int[]{actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        INDArray label = Nd4j.create(new int[]{actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        for (int index = 0; index < actualMiniBatchSize; index++) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            StockPrice curData = train.get(startIdx);
            StockPrice nextData;
            for (int i = startIdx; i < endIdx; i++) {
                nextData = train.get(i + 1);
                int c = i - startIdx;
                input.putScalar(new int[]{index, 0, c}, (curData.getClose() - minNum[0]) / (maxNum[0] - minNum[0]));
                label.putScalar(new int[]{index, 0, c}, (nextData.getClose() - minNum[0]) / (maxNum[0] - minNum[0]));
                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
        return new DataSet(input, label);
    }

    @Override
    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }

    private List<Pair> generateTestDataSet(List<StockPrice> stockPriceList) {
        int window = exampleLength + 1;
        List<Pair> test = new ArrayList<>();
        for (int i = 0; i < stockPriceList.size() - window; i++) {
            INDArray input = Nd4j.create(new int[]{exampleLength, VECTOR_SIZE}, 'f');
            for (int j = i; j < i + exampleLength; j++) {
                StockPrice stock = stockPriceList.get(j);
                input.putScalar(new int[]{j - i, 0}, (stock.getClose() - minNum[0]) / (maxNum[0] - minNum[0]));
            }
            StockPrice stock = stockPriceList.get(i + exampleLength);
            INDArray label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f');
            label.putScalar(new int[]{0}, stock.getClose());
            test.add(new Pair(input, label));
        }
        return test;
    }

    private List<StockPrice> readStockDataFromFile(String filename) {
        List<StockPrice> stockPriceList = new ArrayList<>();
        try {
            @SuppressWarnings("resource")
            List<String[]> list = new CSVReader(new FileReader(filename)).readAll();
            for (int i = 0; i < maxNum.length; i++) {
                maxNum[i] = Double.MIN_VALUE;
                minNum[i] = Double.MAX_VALUE;
            }
            for (String[] arr : list) {
                double[] nums = new double[VECTOR_SIZE];
                for (int i = 0; i < arr.length; i++) {
                    nums[i] = Double.parseDouble(arr[i]);
                    if (nums[i] > maxNum[i])
                        maxNum[i] = nums[i];
                    if (nums[i] < minNum[i])
                        minNum[i] = nums[i];
                }
                stockPriceList.add(new StockPrice(Double.parseDouble(arr[0])));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return stockPriceList;
    }

    @Override
    public int totalExamples() {
        return train.size() - exampleLength - 1;
    }

    @Override
    public int inputColumns() {
        return VECTOR_SIZE;
    }

    @Override
    public int totalOutcomes() {
        return VECTOR_SIZE;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        initializeOffsets();
    }

    @Override
    public int batch() {
        return miniBatchSize;
    }

    @Override
    public int cursor() {
        return totalExamples() - exampleStartOffsets.size();
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not Implemented");
    }
}
