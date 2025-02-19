package com.project.sound.HumanSoundDetection.HumanSoundDetection.service;


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
import java.io.IOException;
import java.io.InputStream;
import java.time.LocalDateTime;
import java.util.Timer;
import java.util.TimerTask;


@Service
public class AudioService {
    private final SoundAnalysisRepository repository;
    private SavedModelBundle model;
    private ConcreteFunction classifyFunction;
    private static final String OUTPUT_FILE = "src/main/resources/speech_model/recorded_audio.wav";
    private TargetDataLine microphone;
    private String prediction="";


    public AudioService(SoundAnalysisRepository repository) {
        this.repository = repository;
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
        float[] mfccFeatures = extractMFCC(buffer);
        prediction = classifySpeech(mfccFeatures);
        saveResult(prediction);
    }

    private float[] extractMFCC(byte[] buffer) {
        float[] audioFeatures = new float[buffer.length];
        for (int i = 0; i < buffer.length; i++) {
            audioFeatures[i] = buffer[i] / 32768.0f; // Normalize to range [-1, 1]
        }
        return audioFeatures;
    }

    private String classifySpeech(float[] mfcc) {
//        try (TFloat32 inputTensor = TFloat32.tensorOf(Shape.of(1, mfcc.length))) {
//            Tensor outputTensor = classifyFunction.call(inputTensor);
//            float prediction = outputTensor.asRawTensor().data().asFloats().getFloat(0);
//
//            return prediction > 0.5 ? "Live Speech Detected" : "Recorded Speech Detected";
//        }
        model = SavedModelBundle.load("src/main/resources/speech_model", "serve");
        try (TFloat32 inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(new float[][]{mfcc}))) {
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
