package com.project.sound.HumanSoundDetection.HumanSoundDetection.service;

import com.jlibrosa.audio.JLibrosa;
import com.project.sound.HumanSoundDetection.HumanSoundDetection.entities.SoundAnalysis;
import com.project.sound.HumanSoundDetection.HumanSoundDetection.repository.SoundAnalysisRepository;
import jakarta.annotation.PreDestroy;
import lombok.SneakyThrows;
import org.springframework.stereotype.Service;
import org.tensorflow.GraphOperation;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.types.TFloat32;

import javax.sound.sampled.*;
import java.io.*;
import java.time.LocalDateTime;
import java.util.Iterator;

@Service
public class AudioService {
    private final SoundAnalysisRepository repository;
    private final MFCCPreprocessor mfccPreprocessor;
    private SavedModelBundle model;

    private static final String MODEL_PATH = "src/main/resources/speech_model";
    private static final String OUTPUT_FILE = "src/main/resources/speech_model/recorded_audio.wav";
    public static final String OUTPUT_FILE_MFCC = "src/main/resources/speech_model/mfcc_features.csv";

    private TargetDataLine microphone;
    private String prediction = "";

    public AudioService(SoundAnalysisRepository repository, MFCCPreprocessor mfccPreprocessor) {
        this.repository = repository;
        this.mfccPreprocessor = mfccPreprocessor;
        // Load model at initialization
        try {
            this.model = SavedModelBundle.load(MODEL_PATH, "serve");
        } catch (Exception e) {
            throw new RuntimeException("Failed to load TensorFlow model", e);
        }
    }

    public String analyzeAudio() {
        try {
            executeRecordingAndAnalysis();
            if (prediction.isEmpty()) {
                return "Error analyzing sound.";
            } else {
                return prediction;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return "Error analyzing sound: " + e.getMessage();
        }
    }

    private void executeRecordingAndAnalysis() throws InterruptedException {
        recordAudio();
        stopRecording();
        waitForFile(OUTPUT_FILE, 5000); // Ensure file is saved before proceeding
        analyzeSoundImplementation();
    }

    @SneakyThrows
    public void recordAudio() {
        // Ensure old file is deleted before starting new recording
        File oldFile = new File(OUTPUT_FILE);
        if (oldFile.exists()) {
            oldFile.delete();
        }

        AudioFormat format = new AudioFormat(16000, 16, 1, true, true);
        DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);

        if (microphone != null) {
            microphone.stop();
            microphone.close();
        }

        microphone = (TargetDataLine) AudioSystem.getLine(info);
        microphone.open(format);
        microphone.start();
        System.out.println("Recording started...");

        File outputFile = new File(OUTPUT_FILE);
        Thread recordingThread = new Thread(() -> {
            try (AudioInputStream audioStream = new AudioInputStream(microphone)) {
                AudioSystem.write(audioStream, AudioFileFormat.Type.WAVE, outputFile);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        recordingThread.start();
        Thread.sleep(20000); // Record for 20 seconds
        stopRecording();
        System.out.println("Recording completed and saved.");
    }

    public void stopRecording() {
        if (microphone != null) {
            microphone.stop();
            microphone.close();
            System.out.println("Microphone closed.");
        }
    }

    private void analyzeSoundImplementation() {
        System.out.println("Analyzing sound...");
        byte[] buffer = loadAudioFile();
        float[] audioData = convertPCMToFloat(buffer);
        float[][] mfccFeatures = extractMFCC(audioData);
        try {
            saveMFCCToCSV(mfccFeatures);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
       // mfccPreprocessor.executeMFCCPreProcessor();

        prediction = classifySpeech(mfccFeatures);
//        saveResult(prediction);
    }
    private String classifySpeech(float[][] mfcc) {

        try (TFloat32 inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(mfcc))) {
            System.out.println("Model operations:");
            for (Iterator<GraphOperation> it = model.graph().operations(); it.hasNext(); ) {
                Operation op = it.next();
                System.out.println(op.name());
            }
            System.out.println("*******************");
            Tensor output = model.session().runner()
                    .feed("serving_default_serving_default_input_1", inputTensor)  // Updated input tensor name
                    .fetch("StatefulPartitionedCall")                              // Updated output tensor name
                    .run()
                    .get(0);
//
//            float prediction = output.asRawTensor().data().asFloats().getFloat(0);
//            return prediction > 0.5 ? "Live Speech" : "Recorded Speech";


            FloatDataBuffer buffer = output.asRawTensor().data().asFloats();
            float first = buffer.getFloat(0);  // live
            float second = buffer.getFloat(1); // recorded

            return first > second ? "Live Speech" : "Recorded Speech";
        } catch (Exception e) {
            throw new RuntimeException("Error during speech classification: " + e.getMessage(), e);
        }
    }
    private float[][] extractMFCC(float[] audioData) {
        JLibrosa jLibrosa = new JLibrosa();
        int sampleRate = 16000;
        int numMFCC = 20;  // Make sure this matches the Python model's input size

        float[][] mfccFeatures = jLibrosa.generateMFCCFeatures(audioData, sampleRate, numMFCC);

        // Average across time frames
        float[] averagedMFCC = new float[numMFCC];
        for (float[] frame : mfccFeatures) {
            for (int i = 0; i < numMFCC; i++) {
                averagedMFCC[i] += frame[i];
            }
        }
        for (int i = 0; i < numMFCC; i++) {
            averagedMFCC[i] /= mfccFeatures.length;
        }
        // Prepare input for model: [1][20]
        float[][] modelInput = new float[1][numMFCC];
        modelInput[0] = averagedMFCC;

        return modelInput;
    }
    private String classifySpeech2(float[][] mfcc) {
        model = SavedModelBundle.load("src/main/resources/speech_model", "serve");
        try (TFloat32 inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(mfcc))) {
            Tensor output = model.session().runner()
                    .feed("input_layer", inputTensor)
                    .fetch("output_layer")
                    .run()
                    .get(0);
            float prediction = output.asRawTensor().data().asFloats().getFloat(0);
            return prediction > 0.5 ? "Live Speech" : "Recorded Speech";
        }
    }

    private void saveResult(String result) {
        repository.save(new SoundAnalysis(result, LocalDateTime.now()));
    }

    private byte[] loadAudioFile() {
        try (InputStream inputStream = new FileInputStream(OUTPUT_FILE)) {
            return inputStream.readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException("Error loading audio file: " + e.getMessage());
        }
    }

    private float[][] extractMFCC2(float[] audioData) {

        JLibrosa jLibrosa = new JLibrosa();
        int sampleRate = 16000;
        int numMFCC = 20; //

        float[][] mfccFeatures = jLibrosa.generateMFCCFeatures(audioData, sampleRate, numMFCC);

        // Average across time frames (if needed)
        float[] averagedMFCC = new float[numMFCC];
        for (float[] frame : mfccFeatures) {
            for (int i = 0; i < numMFCC; i++) {
                averagedMFCC[i] += frame[i];
            }
        }
        for (int i = 0; i < numMFCC; i++) {
            averagedMFCC[i] /= mfccFeatures.length;
        }

        // Prepare input for model: [1][20]
        float[][] modelInput = new float[1][numMFCC];
        modelInput[0] = averagedMFCC;

        return modelInput;
    }

    private void saveMFCCToCSV(float[][] mfccFeatures) throws IOException {
        File oldFile = new File(OUTPUT_FILE_MFCC);
        if (oldFile.exists()) {
            oldFile.delete();
        }
        try {
            Thread.sleep(2000);
        } catch (InterruptedException e) {

        }
        try (FileWriter csvWriter = new FileWriter(OUTPUT_FILE_MFCC)) {
            for (float[] frame : mfccFeatures) {
                for (int j = 0; j < frame.length; j++) {
                    csvWriter.append(String.valueOf(frame[j]));
                    if (j < frame.length - 1) csvWriter.append(",");
                }
                csvWriter.append("\n");
            }
        }
        System.out.println("MFCC saved to CSV.");
    }

    private float[] convertPCMToFloat(byte[] buffer) {
        int sampleCount = buffer.length / 2;
        float[] audioFeatures = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            int low = buffer[2 * i] & 0xFF;
            int high = buffer[2 * i + 1] << 8;
            int sample = high | low;
            audioFeatures[i] = sample / 32768.0f;
        }
        return audioFeatures;
    }

    private void waitForFile(String filename, int timeoutMillis) throws InterruptedException {
        File file = new File(filename);
        int retries = timeoutMillis / 500;
        while (retries-- > 0) {
            if (file.exists() && file.length() > 0) {
                return;
            }
            Thread.sleep(500);
        }
        System.out.println("Warning: File not found or empty after timeout.");
    }
    @PreDestroy
    public void cleanup() {
        if (model != null) {
            model.close();
        }
    }
}
