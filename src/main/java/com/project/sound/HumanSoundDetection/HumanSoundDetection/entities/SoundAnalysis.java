package com.project.sound.HumanSoundDetection.HumanSoundDetection.entities;

import jakarta.persistence.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "sound_analysis")
public class SoundAnalysis {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    public SoundAnalysis() {}


    public Long getId() { return id; }
}
