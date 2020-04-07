package com.mycompany.stockprediction;

import com.mycompany.util.PropertiesUtil;
import com.mycompany.wavelet.WaveletCoefHandler;
import jwave.Transform;
import jwave.transforms.WaveletPacketTransform;
import jwave.transforms.wavelets.Wavelet;
import jwave.transforms.wavelets.daubechies.Daubechies3;
import jwave.transforms.wavelets.haar.Haar1;

import java.util.Arrays;
import java.util.List;

public class WaveletProcessor {

	public static void processor(List<StockData> logList) {
        int len = logList.size();
        int level_decomposed = 3;
        // create a low array, then add to wavelet transform
        double[] openArr = new double[len];
        double[] closeArr = new double[len];
        double[] lowArr = new double[len];
        double[] highArr = new double[len];
        double[] volumeArr = new double[len];
        for(int i=0; i<len; i++)
        {
        	StockData lb = logList.get(i);
        	openArr[i] = lb.getOpen();
        	closeArr[i] = lb.getClose();
        	lowArr[i] = lb.getLow();
        	highArr[i] = lb.getHigh();
        	volumeArr[i]=lb.getVolume();
        }
        
        Wavelet waveletType = null;
        if("DB3".compareToIgnoreCase(PropertiesUtil.getWaveletType())==0)
        	waveletType=new Daubechies3();
        else if("Haar".compareToIgnoreCase(PropertiesUtil.getWaveletType())==0)
        	waveletType=new Haar1();
        
        Transform transform=new Transform(new WaveletPacketTransform(waveletType));
        
        // denoise
        // These values for de-noising the signal only works for 2^p length signals
        double[][] openArrHilb2D = transform.decompose(openArr);
        double[][] closeArrHilb2D = transform.decompose(closeArr);
        double[][] lowArrHilb2D = transform.decompose(lowArr);
        double[][] highArrHilb2D = transform.decompose(highArr);
        double[][] volumeArrHilb2D = transform.decompose(volumeArr);
        
        // calculate the threshold, two methods can be used, use SURE here
//        double threshold = Math.sqrt(2 * Math.log(len * (Math.log(len) / Math.log(2))));
        
        // Get threshold
        double openThreshold = WaveletCoefHandler.getVisushinkThreshold(openArrHilb2D[1]);
        double closeThreshold = WaveletCoefHandler.getVisushinkThreshold(closeArrHilb2D[1]);
        double lowThreshold = WaveletCoefHandler.getVisushinkThreshold(lowArrHilb2D[1]);
        double highThreshold = WaveletCoefHandler.getVisushinkThreshold(highArrHilb2D[1]);
        double volumeThreshold = WaveletCoefHandler.getVisushinkThreshold(volumeArrHilb2D[1]);
        
        
        // Use threshold to denoise
        WaveletCoefHandler.thresholding(openArrHilb2D[level_decomposed], openThreshold, level_decomposed, "hard");
        WaveletCoefHandler.thresholding(closeArrHilb2D[level_decomposed], closeThreshold, level_decomposed, "hard");
        WaveletCoefHandler.thresholding(lowArrHilb2D[level_decomposed], lowThreshold, level_decomposed, "hard");
        WaveletCoefHandler.thresholding(highArrHilb2D[level_decomposed], highThreshold, level_decomposed, "hard");
        WaveletCoefHandler.thresholding(volumeArrHilb2D[level_decomposed], volumeThreshold, level_decomposed, "hard");
        
        
        
        double [] openArrReco = transform.recompose(openArrHilb2D, level_decomposed);
        double [] closeArrReco = transform.recompose(closeArrHilb2D, level_decomposed);
        double [] lowReco = transform.recompose(lowArrHilb2D, level_decomposed);
        double [] highReco = transform.recompose(highArrHilb2D, level_decomposed);
        double [] volumeReco = transform.recompose(volumeArrHilb2D, level_decomposed);
        
        for(int i=0; i<len; i++)
        {
        	StockData lb = logList.get(i);
        	lb.setOpen(openArrReco[i]);
        	lb.setClose(closeArrReco[i]);
        	lb.setLow(lowReco[i]);
        	lb.setHigh(highReco[i]);
        	lb.setVolume(volumeReco[i]);
        }

        System.out.println("open Array Origianl : " + Arrays.toString(openArr));
        System.out.println("open Array Recovered: " + Arrays.toString(openArrReco));
        System.out.println("");
        System.out.println("close Array Origianl : " + Arrays.toString(closeArr));
        System.out.println("close Array Recovered: " + Arrays.toString(closeArrReco));
        System.out.println("");
        System.out.println("low Array Origianl : " + Arrays.toString(lowArr));
        System.out.println("low Array Recovered: " + Arrays.toString(lowReco));
        System.out.println("");
        System.out.println("high Array Origianl : " + Arrays.toString(highArr));
        System.out.println("high Array Recovered: " + Arrays.toString(highReco));
        System.out.println("");
	}
}
