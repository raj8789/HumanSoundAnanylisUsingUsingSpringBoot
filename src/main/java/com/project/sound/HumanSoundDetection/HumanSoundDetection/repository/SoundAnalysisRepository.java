package com.project.sound.HumanSoundDetection.HumanSoundDetection.repository;

import com.project.sound.HumanSoundDetection.HumanSoundDetection.entities.SoundAnalysis;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface SoundAnalysisRepository extends JpaRepository<SoundAnalysis, Long> {}
