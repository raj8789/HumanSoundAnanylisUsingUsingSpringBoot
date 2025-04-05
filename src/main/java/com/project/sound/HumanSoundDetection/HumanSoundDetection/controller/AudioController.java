package com.project.sound.HumanSoundDetection.HumanSoundDetection.controller;

import com.project.sound.HumanSoundDetection.HumanSoundDetection.service.AudioService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/sound")
public class AudioController {
    private AudioService audioService;

    public AudioController(AudioService audioService) {
        this.audioService = audioService;
    }

    @GetMapping("/create/{soundType}")
    public String analyzeSound(@PathVariable String soundType) {
        audioService.soundType=soundType;
        return audioService.analyzeAudio();
    }
}
