# Surf & Strand Analytics - IoT Edge Sensor & Dashboard

Dit project is ontwikkeld als onderdeel van een postgraduaat Big Data & IoT. Het extraheert live data uit een publieke strand-webcam om drukte, weersomstandigheden en de surfkwaliteit (golffrequentie) te monitoren. 

Het project bestaat uit twee gekoppelde systemen:
1. **Sensor Node (`main.py`):** Maakt gebruik van OpenCV, YOLOv8s en adaptieve drempelwaarden om de videostream te analyseren.
2. **Data Dashboard (`dashboard.py`):** Een Streamlit web-applicatie die de ruwe CSV-data interpreteert en visualiseert.

---

## Vereisten (Prerequisites)

Zorg ervoor dat de volgende software op je systeem is geïnstalleerd:
* **Python 3.9+**
* **Node.js:** Noodzakelijk voor de `yt-dlp` bibliotheek om de YouTube JavaScript-beveiliging van livestreams te overbruggen. Download via [nodejs.org](https://nodejs.org/).

---

## Installatieprocedure

Volg deze stappen om het project lokaal te installeren en uit te voeren:

**1. Clone de repository:**
```bash
git clone https://github.com/MathiasUA/SurfAnalyser2.0.git
cd SurfAnalyser2.0
