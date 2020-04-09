package com.mycompany.wavelet;

import com.mycompany.stockprediction.StockData;
import jwave.Transform;
import jwave.transforms.WaveletPacketTransform;
import jwave.transforms.wavelets.Wavelet;
import jwave.transforms.wavelets.haar.Haar1;

import java.util.Arrays;
import java.util.List;

public class WaveletProcessor {

    public static void processor(List<StockData> logList) {
        int len = logList.size();
        int level_decomposed = 3;
        // create a low array, then add to wavelet transform
        double[] closeArr = new double[len];
        for (int i = 0; i < len; i++) {
            StockData lb = logList.get(i);
            closeArr[i] = lb.getClose();
        }

        Wavelet waveletType = new Haar1();

        Transform transform = new Transform(new WaveletPacketTransform(waveletType));

        // denoise
        // These values for de-noising the signal only works for 2^p length signals
        double[][] closeArrHilb2D = transform.decompose(closeArr);

        // calculate the threshold, two methods can be used, use SURE here
//        double threshold = Math.sqrt(2 * Math.log(len * (Math.log(len) / Math.log(2))));

        // Get threshold
        double closeThreshold = WaveletCoefHandler.getVisushinkThreshold(closeArrHilb2D[1]);

        // Use threshold to denoise
        WaveletCoefHandler.thresholding(closeArrHilb2D[level_decomposed], closeThreshold, level_decomposed, "hard");
        double[] closeArrReco = transform.recompose(closeArrHilb2D, level_decomposed);
        for (int i = 0; i < len; i++) {
            StockData lb = logList.get(i);
            lb.setClose(closeArrReco[i]);
        }

        System.out.println("close Array Origianl : " + Arrays.toString(closeArr));
        System.out.println("close Array Recovered: " + Arrays.toString(closeArrReco));
        System.out.println("");
    }
}
