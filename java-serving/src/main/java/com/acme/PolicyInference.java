package com.acme;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.*;
import ai.djl.inference.Predictor;
import ai.djl.translate.*;
import java.nio.file.Paths;

public class PolicyInference {
    static class TranslatorImpl implements Translator<float[], int[]> {
        @Override public NDList process(TranslatorContext ctx, float[] input) {
            NDManager nd = ctx.getNDManager();
            NDArray obs = nd.create(input, new long[]{1, 6}); // [1,6] float32
            return new NDList(obs);
        }
        @Override public int[] processOutput(TranslatorContext ctx, NDList list) {
            NDArray action = list.get(0); // TorchScript returns action tensor [1]
            return new int[]{ (int) action.toType(ai.djl.ndarray.types.DataType.INT64, false).getLong() };
        }
    }

    public static void main(String[] args) throws Exception {
        String modelPath = "outputs/models/model.pt"; // from export_policy.py
        try (Model model = Model.newInstance("ppo_policy", Device.cpu())) {
            model.load(Paths.get(modelPath));
            try (Predictor<float[], int[]> predictor = model.newPredictor(new TranslatorImpl())) {
                float[] obs = new float[]{ 1.2f, 0.6f, 0.9f, 0.3f, 0.08f, 0.06f };
                int[] action = predictor.predict(obs);
                System.out.println("Action: " + action[0]); // 0..4
            }
        }
    }
}
