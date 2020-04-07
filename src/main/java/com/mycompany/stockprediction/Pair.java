package com.mycompany.stockprediction;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Pair {
    private INDArray key;
    private INDArray value;

    public Pair(INDArray key, INDArray value) {
        this.key = key;
        this.value = value;
    }

    public INDArray getKey() {
        return key;
    }

    public INDArray getValue() {
        return value;
    }
}
