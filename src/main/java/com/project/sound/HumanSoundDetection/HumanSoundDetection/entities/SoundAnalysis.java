package com.project.sound.HumanSoundDetection.HumanSoundDetection.entities;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;

import java.time.LocalDateTime;

@Entity
public class SoundAnalysis {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String result;
    private LocalDateTime timestamp;

    public SoundAnalysis() {}

    public SoundAnalysis(String result, LocalDateTime timestamp) {
        this.result = result;
        this.timestamp = timestamp;
    }

    public Long getId() { return id; }
    public String getResult() { return result; }
    public LocalDateTime getTimestamp() { return timestamp; }
}
