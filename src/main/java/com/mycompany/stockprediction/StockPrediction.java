package com.mycompany.stockprediction;

import com.mycompany.network.LSTMNetwork;
import com.mycompany.util.PropertiesUtil;
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

/**
 * NY Stock Exchange prediction
 */
public class StockPrediction {

    public static void main(String[] args) throws IOException {
        System.out.println("Application is starting!");

        int batchSize = PropertiesUtil.getBatchSize();
        int exampleLength = PropertiesUtil.getExampleLength();
        int firstTestItemNumber = PropertiesUtil.getFirstTestItemNumber();
        int testItems = PropertiesUtil.getTestItems();
        String symbol = PropertiesUtil.getStockName(); // stock name

        String file = new ClassPathResource(PropertiesUtil.getDatasetFilename()).getFile().getAbsolutePath();

        System.out.println("Create dataSet iterator...");
        StockDataSetIterator iterator = new StockDataSetIterator(file, symbol, batchSize, exampleLength,
                firstTestItemNumber, testItems);

        System.out.println("Load test dataset...");
        List<Pair> test = iterator.getTestDataSet();

        trainAndTest(iterator, test);
    }

    private static void trainAndTest(StockDataSetIterator iterator, List<Pair> test) throws IOException {
        System.out.println("Build lstm networks...");
        String fileName = "StockLSTM_" + PropertiesUtil.getWaveletType() + ".zip";
        File locationToSave = new File("savedModels/" + fileName);
        MultiLayerNetwork net = LSTMNetwork.buildLSTMNetwork(iterator.inputColumns(), iterator.totalOutcomes());
        // if not use saved model, train new model
        if (PropertiesUtil.getUseSavedModel()) {
            System.out.println("starting to train LSTM networks with " + PropertiesUtil.getWaveletType() + " wavelet...");
            for (int i = 0; i < PropertiesUtil.getEpochs(); i++) {
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
        test(net, test, iterator, PropertiesUtil.getExampleLength(), PropertiesUtil.getEpochs());

        System.out.println("Both the training and testing are finished, system is exiting...");
        System.exit(0);

    }

    private static void test(MultiLayerNetwork net, List<Pair> test, StockDataSetIterator iterator,
                             int exampleLength, int epochNum) {
        System.out.println("Testing...");
        INDArray max = Nd4j.create(iterator.getMaxNum());
        INDArray min = Nd4j.create(iterator.getMinNum());
        INDArray[] predicts = new INDArray[test.size()];
        INDArray[] actuals = new INDArray[test.size()];

        double[] mseValue = new double[PropertiesUtil.getVectorSize()];

        for (int i = 0; i < test.size(); i++) {
            predicts[i] = net.rnnTimeStep(test.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = test.get(i).getValue();
            // Calculate the MSE of results
            mseValue[0] += Math.pow((actuals[i].getDouble(0, 0) - predicts[i].getDouble(0, 0)), 2);
            mseValue[1] += Math.pow((actuals[i].getDouble(0, 1) - predicts[i].getDouble(0, 1)), 2);
            mseValue[2] += Math.pow((actuals[i].getDouble(0, 2) - predicts[i].getDouble(0, 2)), 2);
            mseValue[3] += Math.pow((actuals[i].getDouble(0, 3) - predicts[i].getDouble(0, 3)), 2);
//			mseValue[4] += Math.pow((actuals[i].getDouble(0, 4) - predicts[i].getDouble(0, 4)), 2);
        }

        double mseOpen = Math.sqrt(mseValue[0] / test.size());
        double mseClose = Math.sqrt(mseValue[1] / test.size());
        double mseLow = Math.sqrt(mseValue[2] / test.size());
        double mseHigh = Math.sqrt(mseValue[3] / test.size());
//		double mseVOLUME = Math.sqrt(mseValue[4] / test.size());
//		System.out.println("MSE for [Open,Close,Low,High,VOLUME] is: [" + mseOpen + ", " + mseClose + ", " + mseLow + ", "
//				+ mseHigh + ", " + mseVOLUME);
        System.out.println("MSE for [Open,Close,Low,High] is: [" + mseOpen + ", " + mseClose + ", " + mseLow + ", " + mseHigh + "]");

        // plot predicts and actual values
        System.out.println("Starting to print out values.");
        for (int i = 0; i < predicts.length; i++) {
            System.out.println("Prediction=" + predicts[i] + ", Actual=" + actuals[i]);
        }
        System.out.println("Drawing chart...");
        plotAll(predicts, actuals, epochNum);
        System.out.println("Finished drawing...");
    }

    /**
     * plot all predictions
     *
     * @param predicts
     * @param actuals
     * @param epochNum
     */
    private static void plotAll(INDArray[] predicts, INDArray[] actuals, int epochNum) {
        String STRING_ARRAY_SAMPLE = "E:\\dev\\code\\StockPricePrediction\\savedModels\\result.csv";
        try (Writer writer = Files.newBufferedWriter(Paths.get(STRING_ARRAY_SAMPLE));
                CSVWriter csvWriter = new CSVWriter(writer,
                        CSVWriter.DEFAULT_SEPARATOR,
                        CSVWriter.NO_QUOTE_CHARACTER,
                        CSVWriter.DEFAULT_ESCAPE_CHARACTER,
                        CSVWriter.DEFAULT_LINE_END);) {

		String[] titles = { "PredictOpen", "ActualOpen", "PredictClose", "ActualClose", "PredictLow", "ActualLow", "PredictHigh", "ActualHigh"};
            //String[] titles = {"Open", "Close", "Low", "High"};
//            for (int n = 0; n < PropertiesUtil.getVectorSize(); n++) {
//                double[] pred = new double[predicts.length];
//                double[] actu = new double[actuals.length];
//                for (int i = 0; i < predicts.length; i++) {
//                    pred[i] = predicts[i].getDouble(n);
//                    actu[i] = actuals[i].getDouble(n);
//                    csvWriter.writeNext(new String[]{String.valueOf(pred[i]), String.valueOf(actu[i])});
//                }
//            }
            csvWriter.writeNext(titles);
                for (int i = 0; i < predicts.length; i++) {
                    String predictOpen = String.valueOf(predicts[i].getDouble(0));
                    String predictClose = String.valueOf(predicts[i].getDouble(1));
                    String predictLow = String.valueOf(predicts[i].getDouble(2));
                    String predictHigh = String.valueOf(predicts[i].getDouble(3));

                    String actualOpen = String.valueOf(actuals[i].getDouble(0));
                    String actualClose = String.valueOf(actuals[i].getDouble(1));
                    String actualLow = String.valueOf(actuals[i].getDouble(2));
                    String actualHigh = String.valueOf(actuals[i].getDouble(3));
                    csvWriter.writeNext(new String[]{predictOpen, actualOpen, predictClose, actualClose, predictLow, actualLow, predictHigh, actualHigh});
                }
            //DrawingTool.drawChart(pred, actu, titles[n], epochNum);
        } catch (IOException e) {
			e.printStackTrace();
		}
	}


}
