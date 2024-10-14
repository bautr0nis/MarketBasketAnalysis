import hex.genmodel.GenModel;
import hex.genmodel.MojoModel;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.RowData;
import hex.genmodel.easy.prediction.BinomialModelPrediction;

public class MojoExample {
    public static void main(String[] args) throws Exception {
        // Path to your MOJO model file
        String mojoFilePath = "GBM_model_python_1728838744377_1869.zip";

        // Load the MOJO model
        MojoModel mojo = MojoModel.load(mojoFilePath);
        EasyPredictModelWrapper model = new EasyPredictModelWrapper(mojo);

        // Create a sample input row
        RowData row = new RowData();
        row.put("feature1", "value1");
        row.put("feature2", "value2");
        // Add other features as needed

        // Generate prediction
        BinomialModelPrediction prediction = model.predictBinomial(row);
        System.out.println("Predicted class: " + prediction.label);
        System.out.println("Probability for class 0: " + prediction.classProbabilities[0]);
        System.out.println("Probability for class 1: " + prediction.classProbabilities[1]);
    }
}
