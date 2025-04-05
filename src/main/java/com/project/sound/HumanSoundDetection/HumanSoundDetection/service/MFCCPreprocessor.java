package com.project.sound.HumanSoundDetection.HumanSoundDetection.service;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.springframework.stereotype.Service;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Service
public class MFCCPreprocessor {

    public void executeMFCCPreProcessor(){
       // String filePath = "mfcc_features.csv"; // Update your file path
        List<double[]> featureList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();
        //  Load CSV File
        try (Reader reader = Files.newBufferedReader(Paths.get(AudioService.OUTPUT_FILE_MFCC+AudioService.count+AudioService.CSV_FILE_NAME));
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT)) {
            for (CSVRecord record : csvParser) {
                int numColumns = record.size();
                double[] features = new double[numColumns - 1];  // All except last column
                String label = record.get(numColumns - 1);      // Last column is the label
                // Convert features to double
                for (int i = 0; i < numColumns - 1; i++) {
                    features[i] = Double.parseDouble(record.get(i));
                }
                //  Encode Labels (Live = 0, Recorded = 1)
                int encodedLabel = label.equalsIgnoreCase("live") ? 0 : 1;

                featureList.add(features);
                labelList.add(encodedLabel);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // Convert Lists to Arrays
        double[][] featuresArray = featureList.toArray(new double[0][]);
        int[] labelsArray = labelList.stream().mapToInt(i -> i).toArray();
        //  Normalize Features
        normalize(featuresArray);
        //  Split into Train & Test (80-20)
        int splitIndex = (int) (0.8 * featuresArray.length);
        List<double[]> trainFeatures = Arrays.asList(Arrays.copyOfRange(featuresArray, 0, splitIndex));
        List<double[]> testFeatures = Arrays.asList(Arrays.copyOfRange(featuresArray, splitIndex, featuresArray.length));
        List<Integer> trainLabels = labelList.subList(0, splitIndex);
        List<Integer> testLabels = labelList.subList(splitIndex, labelList.size());

        // Save processed data
        saveToFile( AudioService.OUTPUT_FILE_TEST_TRAIN+"/X_train.txt", trainFeatures);
        saveToFile(AudioService.OUTPUT_FILE_TEST_TRAIN+"/X_test.txt", testFeatures);
        saveToFile(AudioService.OUTPUT_FILE_TEST_TRAIN+"/y_train.txt", trainLabels);
        saveToFile(AudioService.OUTPUT_FILE_TEST_TRAIN+"/y_test.txt", testLabels);

        System.out.println("Preprocessing Completed! Train/Test Data Saved.");
    }
    // 🔹 Normalize Features using Mean & Standard Deviation
    private  void normalize(double[][] data) {
        int numFeatures = data[0].length;
        Mean meanCalc = new Mean();
        StandardDeviation stdCalc = new StandardDeviation();
        for (int i = 0; i < numFeatures; i++) {
            final int index = i;
            double[] columnData = Arrays.stream(data).mapToDouble(row -> row[index]).toArray();
            double mean = meanCalc.evaluate(columnData);
            double std = stdCalc.evaluate(columnData);
            for (double[] row : data) {
                row[i] = (row[i] - mean) / (std + 1e-8); // Normalize
            }
        }
    }
    // 🔹 Save Data to File
    private  void saveToFile(String filename, List<?> data) {
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get(filename))) {
            for (Object row : data) {
                writer.write(row.toString());
                writer.newLine();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
