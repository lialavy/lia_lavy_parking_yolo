# Parking Space Detection System

## Setup Instructions

1. **Clone repository**
   ```
   git clone https://github.com/lialavy/lia_lavy_parking_yolo.git
   cd lia_lavy_parking_yolo
   ```

2. **Set up environment**
   ```
   python -m venv venv
   venv\Scripts\activate  # Windows
   #source venv/bin/activate   Mac/Linux
   pip install -r requirements.txt
   ```

3. **Download required files**
   - Download the YOLO model file (best (4).pt)
   - Download sample videos (parkinglot1.mp4,parkinglot2.mp4,parkinglot1.mp4_snapshot.jpeg etc... even add your own videos and images!)
   

4. **Run the application**
   ```
   python app.py
   ```
   Access at: http://127.0.0.1:5000/

5. **Create account**
   - Sign up with any username/password
   - For admin access: change email: lia.ninio24@gmail.com to your own email

## Usage

- **Dashboard**: Shows live parking detection with occupied/available counts
- **History**: View parking space status changes over time

## Admin Functions

- **Add parking lot**: Specify video source, model, and bounding boxes file
- **Edit settings**: Modify existing parking lot configurations

## IMPORTANT: Path Configuration

**When adding or editing a parking lot, you MUST use the full absolute path to files, not just the filename:**

- **CORRECT**: `C:/Users/YourName/lia_lavy_parking_yolo/best (4).pt`
- **INCORRECT**: `best (4).pt`

This applies to:
- Model Path
- Video Source
- Bounding Boxes File

Example for Windows:
```
Model Path: C:/Users/YourName/lia_lavy_parking_yolo/best (4).pt
Video Source: C:/Users/YourName/lia_lavy_parking_yolo/carPark.mp4
Bounding Boxes File: C:/Users/YourName/lia_lavy_parking_yolo/bounding_boxes.json
```

Use forward slashes (/) in paths, even on Windows.

## Troubleshooting

- If errors occur with JSON files, ensure they use UTF-8 encoding
- For slow performance, edit app.py to use CPU: `DEVICE = "cpu"`
- Make sure model file (best (4).pt) exists in the project directory
- Check video paths in settings match actual file locations
- If you get path errors, verify you're using complete absolute paths
