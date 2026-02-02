# Sport Multi-Person Tracker (GitHub Pages)

Live-Demo: GitHub Pages (Frontend only). Tracking läuft vollständig im Browser.

## Features
- Multi-Person Pose Tracking (MediaPipe PoseLandmarker)
- Track-IDs (heuristisch, nearest-neighbor)
- Team-Zuordnung über Leibchenfarbe (Torso-ROI)
- Feld-Kalibrierung über 4 farbige Marker (Homographie) + Mini-Map

## Setup (GitHub Pages)
1. Repo → Settings → Pages  
2. Deploy from branch: `main`, Folder: `/docs`
3. Öffne die erzeugte URL: `https://<user>.github.io/<repo>/`

## Praxis-Tipps
- 4 unterschiedlich farbige Hütchen in die Ecken (starker Kontrast!)
- Stativkamera erhöht Tracking-Stabilität
- Teamfarben möglichst weit auseinander (Neon vs. Neon)

## Lizenz
Dieses Projekt nutzt MediaPipe (Apache-2.0). Verarbeitung lokal im Browser. Keine Server.
