package com.mycompany.stockprediction;

import com.mycompany.network.LSTMNetwork;
import com.mycompany.util.DrawingTool;
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
    private static final int epochs = 50;
    private static final String datasetFilename = "AAP.csv";
    private static final int firstTestItemNumber = 500;
    private static final int testItems = 250;
    private static final boolean useSavedModel = false;

    public static void main(String[] args) throws IOException {
        String file = new ClassPathResource(datasetFilename).getFile().getAbsolutePath();

        StockDataSetIterator iterator = new StockDataSetIterator(file, batchSize, exampleLength,
                firstTestItemNumber, testItems);
        List<Pair> test = iterator.getTestDataSet();
        trainAndTest(iterator, test);
    }

    private static void trainAndTest(StockDataSetIterator iterator, List<Pair> test) throws IOException {
        System.out.println("Build lstm networks...");
        File locationToSave = new File("src/main/resources/savedModel/StockLSTM.zip");
        MultiLayerNetwork net = LSTMNetwork.buildLSTMNetwork(iterator.inputColumns(), iterator.totalOutcomes());

        if (!useSavedModel) {
            System.out.println("starting to train LSTM networks");
            for (int i = 0; i < epochs; i++) {
                System.out.println("training at epoch " + i);
                DataSet dataSet;
                while (iterator.hasNext()) {
                    dataSet = iterator.next();
                    net.fit(dataSet);
                }
                iterator.reset();
                net.rnnClearPreviousState();
            }
            System.out.println("saving trained network model...");
            ModelSerializer.writeModel(net, locationToSave, true);
        } else {
            System.out.println("loading network model...");
            net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
        }
        test(net, test, iterator);

    }

    private static void test(MultiLayerNetwork net, List<Pair> test, StockDataSetIterator iterator) {
        System.out.println("Testing...");
        INDArray max = Nd4j.create(iterator.getMaxNum());
        INDArray min = Nd4j.create(iterator.getMinNum());
        INDArray[] predicts = new INDArray[test.size()];
        INDArray[] actuals = new INDArray[test.size()];

        for (int i = 0; i < test.size(); i++) {
            predicts[i] = net.rnnTimeStep(test.get(i).getKey()).getRow(StockPrediction.exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = test.get(i).getValue();
        }

        for (int i = 0; i < predicts.length; i++) {
            System.out.println("Prediction=" + predicts[i] + ", Actual=" + actuals[i]);
        }
        System.out.println("Drawing chart...");
        saveResults(predicts, actuals);
        System.out.println("Finished drawing...");
    }

    private static void saveResults(INDArray[] predicts, INDArray[] actuals) {
        String STRING_ARRAY_SAMPLE = "src/main/resources/result.csv";
        try (Writer writer = Files.newBufferedWriter(Paths.get(STRING_ARRAY_SAMPLE));
             CSVWriter csvWriter = new CSVWriter(writer)) {

            double[] predictDouble = new double[predicts.length];
            double[] actualDouble = new double[actuals.length];
            String[] titles = {"PredictClose", "ActualClose"};
            csvWriter.writeNext(titles);
            for (int i = 0; i < predicts.length; i++) {
                String predictClose = String.valueOf(predicts[i].getDouble(0));
                String actualClose = String.valueOf(actuals[i].getDouble(0));
                csvWriter.writeNext(new String[]{predictClose, actualClose});
                predictDouble[i] = predicts[i].getDouble(0);
                actualDouble[i] = actuals[i].getDouble(0);
            }
            DrawingTool.drawChart(predictDouble, actualDouble);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
