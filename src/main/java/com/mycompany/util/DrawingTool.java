package com.mycompany.util;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;

public class DrawingTool {

    public static void drawChart(double[] predicts, double[] actuals) {
        double[] dataIndex = new double[predicts.length];
        for (int i = 0; i < predicts.length; i++) {
            dataIndex[i] = i;
        }
        double max = Arrays.stream(predicts).max().getAsDouble();
        ;
        double min = Arrays.stream(predicts).min().getAsDouble();
        ;

        XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet, dataIndex, predicts, "Predicts");
        addSeries(dataSet, dataIndex, actuals, "Actuals");
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Close prise",
                "Data Index",
                "Value",
                dataSet,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        XYPlot xyPlot = chart.getXYPlot();

        final NumberAxis rangeAxis = (NumberAxis) xyPlot.getRangeAxis();
        rangeAxis.setRange(0.9 * min, max * 1.2);

        final ChartPanel panel = new ChartPanel(chart);
        final JFrame jFrame = new JFrame();
        jFrame.add(panel);
        jFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        jFrame.pack();
        jFrame.setVisible(true);
        saveChartToPicture(jFrame);
    }

    private static void saveChartToPicture(JFrame jFrame) {
        BufferedImage bi = new BufferedImage(jFrame.getWidth(), jFrame.getHeight(), BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2d = bi.createGraphics();
        jFrame.paint(g2d);
        try {
            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH-mm-ss");
            ImageIO.write(bi, "PNG", new File("src/main/resources/charts/"
                    + LocalDateTime.now().format(formatter) + ".png"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void addSeries(final XYSeriesCollection dataSet, double[] x, double[] y, final String label) {
        final XYSeries s = new XYSeries(label);
        for (int j = 0; j < x.length; j++) {
            s.add(x[j], y[j]);
        }
        dataSet.addSeries(s);
    }
}
