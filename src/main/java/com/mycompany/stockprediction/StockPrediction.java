package com.mycompany.stockprediction;

import com.mycompany.network.LSTMNetwork;
import com.opencsv.CSVWriter;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class StockPrediction {

    private static final int batchSize = 64;
    private static final int exampleLength = 32;
    private static final int epochs = 60;
    private static final int vectorSize = 4;
    private static final String datasetFilename = "NY-StockExchange.csv";
    private static final String stockName = "AAPL";
    private static final int firstTestItemNumber = 1024;
    private static final int testItems = 500;
    private static final boolean useSavedModel = false;

    public static void main(String[] args) throws IOException {
        System.out.println("Application is starting!");

        String file = new ClassPathResource(datasetFilename).getFile().getAbsolutePath();

        System.out.println("Create dataSet iterator...");
        StockDataSetIterator iterator = new StockDataSetIterator(file, stockName, batchSize, exampleLength,
                firstTestItemNumber, testItems);

        System.out.println("Load test dataset...");
        List<Pair> test = iterator.getTestDataSet();

        trainAndTest(iterator, test);
    }

    private static void trainAndTest(StockDataSetIterator iterator, List<Pair> test) throws IOException {
        System.out.println("Build lstm networks...");
        String fileName = "StockLSTM_Haar.zip";
        File locationToSave = new File("savedModels/" + fileName);
        MultiLayerNetwork net = LSTMNetwork.buildLSTMNetwork(iterator.inputColumns(), iterator.totalOutcomes());
        // if not use saved model, train new model
        if (useSavedModel) {
            System.out.println("starting to train LSTM networks with Haar wavelet...");
            for (int i = 0; i < epochs; i++) {
                System.out.println("training at epoch " + i);
                DataSet dataSet;
                while (iterator.hasNext()) {
                    dataSet = iterator.next();
                    net.fit(dataSet);
                }
                iterator.reset(); // reset iterator
                net.rnnClearPreviousState(); // clear previous state
            }
            // save model to file
            System.out.println("saving trained network model...");
            ModelSerializer.writeModel(net, locationToSave, true);
        } else {
            System.out.println("loading network model...");
            net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        }
        // testing
        test(net, test, iterator);

        System.out.println("Both the training and testing are finished, system is exiting...");
        System.exit(0);

    }

    private static void test(MultiLayerNetwork net, List<Pair> test, StockDataSetIterator iterator) {
        System.out.println("Testing...");
        INDArray max = Nd4j.create(iterator.getMaxNum());
        INDArray min = Nd4j.create(iterator.getMinNum());
        INDArray[] predicts = new INDArray[test.size()];
        INDArray[] actuals = new INDArray[test.size()];

        double[] mseValue = new double[vectorSize];

        for (int i = 0; i < test.size(); i++) {
            predicts[i] = net.rnnTimeStep(test.get(i).getKey()).getRow(StockPrediction.exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = test.get(i).getValue();
            // Calculate the MSE of results
            mseValue[1] += Math.pow((actuals[i].getDouble(0, 0) - predicts[i].getDouble(0, 0)), 2);
        }

        double mseClose = Math.sqrt(mseValue[1] / test.size());
        System.out.println("Starting to print out values.");
        for (int i = 0; i < predicts.length; i++) {
            System.out.println("Prediction=" + predicts[i] + ", Actual=" + actuals[i]);
        }
        System.out.println("Drawing chart...");
        plotAll(predicts, actuals);
        System.out.println("Finished drawing...");
    }

    private static void plotAll(INDArray[] predicts, INDArray[] actuals) {
        String STRING_ARRAY_SAMPLE = "E:\\dev\\code\\StockPricePrediction\\savedModels\\result.csv";
        try (Writer writer = Files.newBufferedWriter(Paths.get(STRING_ARRAY_SAMPLE));
             CSVWriter csvWriter = new CSVWriter(writer,
                     CSVWriter.DEFAULT_SEPARATOR,
                     CSVWriter.NO_QUOTE_CHARACTER,
                     CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                     CSVWriter.DEFAULT_LINE_END);) {

            String[] titles = {"PredictClose", "ActualClose"};
            csvWriter.writeNext(titles);
            for (int i = 0; i < predicts.length; i++) {
                String predictClose = String.valueOf(predicts[i].getDouble(0));
                String actualClose = String.valueOf(actuals[i].getDouble(0));
                csvWriter.writeNext(new String[]{predictClose, actualClose});
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
