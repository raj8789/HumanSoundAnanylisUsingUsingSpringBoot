package com.project.sound.HumanSoundDetection.HumanSoundDetection.service;


import com.jlibrosa.audio.JLibrosa;
import com.project.sound.HumanSoundDetection.HumanSoundDetection.entities.SoundAnalysis;
import com.project.sound.HumanSoundDetection.HumanSoundDetection.repository.SoundAnalysisRepository;
import lombok.SneakyThrows;
import org.springframework.stereotype.Service;
import org.tensorflow.ConcreteFunction;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;

import javax.sound.sampled.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.time.LocalDateTime;
import java.util.Timer;
import java.util.TimerTask;


@Service
public class AudioService {
    private final SoundAnalysisRepository repository;
    private final  MFCCPreprocessor mfccPreprocessor;
    private SavedModelBundle model;
    private ConcreteFunction classifyFunction;
    private static final String OUTPUT_FILE = "src/main/resources/speech_model/recorded_audio.wav";

    public static final String OUTPUT_FILE_MFCC="src/main/resources/speech_model/mfcc_features.csv";

    private TargetDataLine microphone;
    private String prediction="";


    public AudioService(SoundAnalysisRepository repository,MFCCPreprocessor mfccPreprocessor) {
        this.repository = repository;
        this.mfccPreprocessor=mfccPreprocessor;
    }

    private byte[] loadModel() {
        String filename="speech_model/recorded_audio.wav";
        try (InputStream inputStream = Thread.currentThread().getContextClassLoader().getResourceAsStream(filename)) {
            if (inputStream == null) throw new IOException("File not found: " + filename);
            return inputStream.readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }
    private void scheduleTaskStartSoundRecording(){
        Timer timer = new Timer();
        // Schedule a task to run after 1 seconds (1000 milliseconds)
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                System.out.println("Thread executed after 1 seconds: " + Thread.currentThread().getName());
                recordAudio();
                timer.cancel(); // Stop the timer after execution
            }
        }, 1000);
    }

    public String analyzeAudio() {
        try {
            scheduledTask();
            if(prediction.isEmpty()){
                return "Error analyzing sound: ";
            }else{
                return prediction;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return "Error analyzing sound: " + e.getMessage();
        }
    }
    private void scheduleTaskToStopRecording(){
        Timer timer = new Timer();
        // Schedule a task to run after 15 seconds (15000 milliseconds)
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                System.out.println("Thread executed after 15 seconds: " + Thread.currentThread().getName());
                stopRecording();
                System.out.println("Executing task For Stop Recoding after 15 seconds!");
                timer.cancel(); // Stop the timer after execution
            }
        }, 15000);
    }
    private void scheduledTask(){
        scheduleTaskToAnalyzeSound();
        scheduleTaskToStopRecording();
        scheduleTaskStartSoundRecording();
    }
    private void scheduleTaskToAnalyzeSound(){
        Timer timer = new Timer();
        // Schedule a task to run after 15 seconds (15000 milliseconds)
        timer.schedule(new TimerTask() {
            @Override
            public void run() {
                System.out.println("Thread executed after 20 seconds: " + Thread.currentThread().getName());
                System.out.println("Executing task For Analyzing Sound after 20 seconds!");
                analyzeSoundImplementation();
                timer.cancel(); // Stop the timer after execution
            }
        }, 20000);
    }
    private void analyzeSoundImplementation(){
        byte[] buffer = loadModel();
//        microphone.read(buffer, 0, buffer.length);
//        microphone.close();

        // Machine Learning Classification
        float[] audioData = convertPCMToFloat(buffer);
        float[][] mfccFeatures=extractMFCC(audioData);
        try {
            saveMFCCToCSV(mfccFeatures);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        mfccPreprocessor.executeMFCCPreProcessor();
//        prediction = classifySpeech(mfccFeatures);
//        saveResult(prediction);
    }

    private float[][] extractMFCC(float[] audioData) {
        // Initialize JLibrosa
        JLibrosa jLibrosa = new JLibrosa();
        // Extract MFCC features
        int sampleRate = 16000; // Adjust according to your dataset
        int numMFCC = 13; // Typical number of MFCC coefficients
        float[][] mfccFeatures = jLibrosa.generateMFCCFeatures(audioData, sampleRate, numMFCC);
        System.out.println("MFCC Feature Sample:");
        for (int i = 0; i < Math.min(5, mfccFeatures.length); i++) {
            System.out.println(java.util.Arrays.toString(mfccFeatures[i]));
        }
        return mfccFeatures;
    }
    public void saveMFCCToCSV(float[][] mfccFeatures) throws IOException {
        FileWriter csvWriter = new FileWriter(OUTPUT_FILE_MFCC);
        for (float[] frame : mfccFeatures) {
            for (int j = 0; j < frame.length; j++) {
                csvWriter.append(String.valueOf(frame[j]));
                if (j < frame.length - 1) csvWriter.append(",");
            }
            csvWriter.append("\n");
        }
        csvWriter.flush();
        csvWriter.close();
    }
    // Convert 16-bit PCM bytes to float values
    private  float[] convertPCMToFloat(byte[] buffer) {
        int sampleCount = buffer.length / 2; // Each sample is 2 bytes (16-bit PCM)
        float[] audioFeatures = new float[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            int low = buffer[2 * i] & 0xFF; // Least significant byte (LSB)
            int high = buffer[2 * i + 1] << 8; // Most significant byte (MSB)
            int sample = high | low; // Combine bytes to form 16-bit sample
            audioFeatures[i] = sample / 32768.0f; // Normalize to [-1, 1]
        }
        return audioFeatures;
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
    @SneakyThrows
    public void recordAudio() {
        AudioFormat format = new AudioFormat(16000, 16, 2, true, true);
        DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
        microphone = (TargetDataLine) AudioSystem.getLine(info);
        microphone.open(format);
        microphone.start();
        System.out.println("Recording...");
        AudioInputStream audioStream = new AudioInputStream(microphone);
        File outputFile = new File(OUTPUT_FILE);
        AudioSystem.write(audioStream, AudioFileFormat.Type.WAVE, outputFile);
        System.out.println("Recording2...");
    }

    public void stopRecording(){
        microphone.stop();
        microphone.close();
        System.out.println("Recording saved as " + OUTPUT_FILE);
    }
}
