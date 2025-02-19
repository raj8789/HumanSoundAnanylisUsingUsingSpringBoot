package com.project.sound.HumanSoundDetection.HumanSoundDetection.websocket;


import org.springframework.web.socket.*;

import java.io.IOException;
import java.util.concurrent.CopyOnWriteArrayList;

public class SoundWebSocketHandler implements WebSocketHandler {
    private static final CopyOnWriteArrayList<WebSocketSession> sessions = new CopyOnWriteArrayList<>();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        sessions.add(session);
    }

    @Override
    public void handleMessage(WebSocketSession session, WebSocketMessage<?> message) {

    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) {

    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        sessions.remove(session);
    }

    @Override
    public boolean supportsPartialMessages() {
        return false;
    }

    public static void sendNotification(String message) throws IOException {
        for (WebSocketSession session : sessions) {
            session.sendMessage(new TextMessage(message));
        }
    }
}

