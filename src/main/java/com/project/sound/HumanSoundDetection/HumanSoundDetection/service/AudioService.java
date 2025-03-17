package com.project.sound.HumanSoundDetection.HumanSoundDetection.service;

import com.jlibrosa.audio.JLibrosa;
import com.project.sound.HumanSoundDetection.HumanSoundDetection.entities.SoundAnalysis;
import com.project.sound.HumanSoundDetection.HumanSoundDetection.repository.SoundAnalysisRepository;
import lombok.SneakyThrows;
import org.springframework.stereotype.Service;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

import javax.sound.sampled.*;
import java.io.*;
import java.time.LocalDateTime;
import java.util.Timer;
import java.util.TimerTask;

@Service
public class AudioService {
    private final SoundAnalysisRepository repository;
    private final MFCCPreprocessor mfccPreprocessor;
    private SavedModelBundle model;

    private static final String OUTPUT_FILE = "src/main/resources/speech_model/recorded_audio.wav";
    public static final String OUTPUT_FILE_MFCC = "src/main/resources/speech_model/mfcc_features.csv";

    private TargetDataLine microphone;
    private String prediction = "";

    public AudioService(SoundAnalysisRepository repository, MFCCPreprocessor mfccPreprocessor) {
        this.repository = repository;
        this.mfccPreprocessor = mfccPreprocessor;
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

        AudioFormat format = new AudioFormat(16000, 16, 2, true, true);
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
        mfccPreprocessor.executeMFCCPreProcessor();
        //        prediction = classifySpeech(mfccFeatures);
//        saveResult(prediction);
    }
    private String classifySpeech(float[][] mfcc) {
//        try (TFloat32 inputTensor = TFloat32.tensorOf(Shape.of(1, mfcc.length))) {
//            Tensor outputTensor = classifyFunction.call(inputTensor);
//            float prediction = outputTensor.asRawTensor().data().asFloats().getFloat(0);
//
//            return prediction > 0.5 ? "Live Speech Detected" : "Recorded Speech Detected";
//        }
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

    private float[][] extractMFCC(float[] audioData) {
        JLibrosa jLibrosa = new JLibrosa();
        int sampleRate = 16000;
        int numMFCC = 13;
        float[][] mfccFeatures = jLibrosa.generateMFCCFeatures(audioData, sampleRate, numMFCC);
        System.out.println("MFCC Feature Sample:");
        for (int i = 0; i < Math.min(5, mfccFeatures.length); i++) {
            System.out.println(java.util.Arrays.toString(mfccFeatures[i]));
        }
        System.out.println("Extracted MFCC Features.");
        return mfccFeatures;
    }

    private void saveMFCCToCSV(float[][] mfccFeatures) throws IOException {
        File oldFile = new File(OUTPUT_FILE_MFCC);
        if (oldFile.exists()) {
            oldFile.delete();
        }
        try {
            Thread.sleep(1000);
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
}
