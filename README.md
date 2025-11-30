🚗💥 Vehicle Accident Detection System (YOLOv8 + OpenCV + Gradio)

A real-time vehicle accident detection system that uses YOLOv8, OpenCV, and a clean Gradio web interface to detect accidents from:
📁 Uploaded traffic videos
💻 Laptop webcam
📱 Phone IP Webcam (via IP Camera URL)

The system identifies potential accidents using vehicle detection, collision analysis (IoU), and bounding-box area growth, then saves:
📸 Accident frames as JPG
📄 Accident logs as CSV

This project is suitable for Smart India Hackathon (SIH), major projects, ML portfolios, and research work.

⭐ Features
🔍 AI-Based Accident Detection
YOLOv8 for vehicle detection
Collision detection using:
IoU (overlap)
Sudden area growth
Works on any traffic/camera footage

🖥️ Multiple Input Modes
Upload video file
Laptop webcam
Mobile phone camera via IP Webcam app (http://<your_ip>:8080/video)

📸 Automated Evidence Extraction
Accident frames saved as JPG
Time-stamped accident logs saved as CSV

🌐 Modern Web UI (Gradio)
Clean design
Threshold sliders
User-friendly workflow

📂 Project Structure
vehicle-accident-detector/
│
├── app.py                          # Main Gradio launcher
├── requirements.txt                # Dependencies
├── .gitignore                      # Ignore venv, outputs, weights, cache
│
├── detector/
│   ├── model.py                    # YOLO model loader
│   ├── pipeline.py                 # Accident detection pipeline
│   └── utils.py                    # Helper functions
│
├── ui/
│   └── gradio_app.py               # Full Gradio UI (video + webcam + IP cam)
│
├── outputs/                        # Saved JPGs + CSV logs (ignored in Git)
└── README.md                       # Project documentation


Installation (Local Machine)
1️⃣ Clone the repository
git clone https://github.com/Ishaan12344/vehicle-accident-detector.git
cd vehicle-accident-detector

2️⃣ Create virtual environment
python -m venv .venv
Activate it:
.\.venv\Scripts\activate

3️⃣ Install all dependencies
pip install -r requirements.txt

4️⃣ Run the application
python app.py

🧪 Output Files
🖼️ Accident Frames (JPG)
Saved automatically in:
outputs/frames/

📄 Accident Log (CSV)
Saved in:
outputs/logs/
Contains:
| Frame | Timestamp | Vehicle IDs | IoU | Area Growth |

📦 Technologies Used
YOLOv8 (Ultralytics)
OpenCV
Python 3.10+
Gradio (Web UI)
FastAPI / Uvicorn
NumPy
TQDM

 Future Enhancements 
🔵 Vehicle tracking with DeepSORT
🟢 Accident everity classification (minor/major)
🛰️ Drone camera integration
📡 IoT emergency alert system
🚑 Automatic dispatch notification
📊 Monitoring dashboard (Plotly/Streamlit)
☁️ Cloud deployment (Render, HF Spaces)
If you want, I can help you implement any of these features.

👨‍💻 Contributors
Ishaan Khanchandani — Team Lead & Machine Learning Developer
Harsh Daulatani — Data Processing & Testing
Taran Wadhawan — UI/UX & Documentation