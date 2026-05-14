"use client";

import { useState, useRef } from "react";
import "./control-panel.css";

export default function Home() {
  const [ipAddress, setIpAddress] = useState("192.168.1.100");
  const [status, setStatus] = useState("Hazır (Ready)");
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const sendCommand = async (queryString: string, label: string) => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setStatus(`${label} komutu gönderiliyor...`);
    try {
      // url oluştururken doğrudan IP adresinin sonuna queryString ekliyoruz
      // Örn: http://192.168.1.100/?11=90&12=120&13=20&14=90&15=160
      const url = `http://${ipAddress}${queryString}`;

      await fetch(url, { method: "GET", mode: "no-cors" });

      setStatus(`${label} başarıyla gönderildi!`);

      timeoutRef.current = setTimeout(() => {
        setStatus("Hazır (Ready)");
      }, 3000);
    } catch (error) {
      console.error(error);
      setStatus(`Hata: ${label} komutu gönderilemedi.`);
    }
  };

  const handleDropTrash = async () => {
    // 1. İstek (Örn: Çöpü bırakma konumuna gitme)
    await sendCommand("/?11=90&12=120&13=130&14=90&15=60", "Çöpü Bırak (1/2)");

    // 2 saniye bekle
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 2. İstek (Örn: Çöpü bırakmak için gripper'ı açma) - Bu açıları kendi sisteminize göre ayarlayın
    await sendCommand("/?11=10&12=120&13=80&14=90&15=160", "Çöpü Bırak (2/2)");
  };

  return (
    <div className="container">
      <div className="glass-panel">
        <div className="header">
          <div className="status-indicator animate-pulse"></div>
          <h1>Robot Kontrol Merkezi</h1>
        </div>

        <div className="ip-section">
          <label htmlFor="ip-input">Robot IP Adresi</label>
          <input
            id="ip-input"
            type="text"
            value={ipAddress}
            onChange={(e) => setIpAddress(e.target.value)}
            className="ip-input"
            placeholder="Örn: 192.168.1.100"
          />
        </div>

        <div className="buttons-grid">
          <button
            // Buradaki tırnak içindeki değerleri kendi motor açılarınıza göre değiştirebilirsiniz
            onClick={() => sendCommand("/?11=90&12=120&13=20&14=90&15=160", "Bekleme Konumu")}
            className="btn btn-primary"
          >
            <span className="btn-icon">⏳</span>
            Bekleme Konumu
          </button>

          <button
            // Çöpü Al için gereken açıları buraya yazabilirsiniz
            onClick={() => sendCommand("/?11=90&12=70&13=130&14=90&15=60", "Çöpü Al")}
            className="btn btn-success"
          >
            <span className="btn-icon">🦾</span>
            Çöpü Al
          </button>

          <button
            // Çöpü Bırak için iki aşamalı istek atılır
            onClick={handleDropTrash}
            className="btn btn-warning"
          >
            <span className="btn-icon">🗑️</span>
            Çöpü Bırak
          </button>
        </div>

        <div className="status-bar">
          <span className="status-label">Sistem Durumu:</span>
          <span className="status-text">{status}</span>
        </div>
      </div>
    </div>
  );
}
