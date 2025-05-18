I understand. Let me create a more direct and user-friendly tutorial:

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
   source venv/bin/activate  # Mac/Linux
   pip install -r requirements.txt
   ```

3. **Download required files**
   - Download the YOLO model file (best (4).pt)
   - Download sample videos (carPark.mp4)
   - Place in the root directory

4. **Run the application**
   ```
   python app.py
   ```
   Access at: http://127.0.0.1:5000/

5. **Create account**
   - Sign up with any username/password
   - For admin access: change the email: lia.ninio24@gmail.com to your email

## Usage

- **Dashboard**: Shows live parking detection with occupied/available counts
- **History**: View parking space status changes over time

## Admin Functions

- **Add parking lot**: Specify video source, model, and bounding boxes file
- **Edit settings**: Modify existing parking lot configurations
- **Path format**: Use filenames for included files or full paths for external files

## Troubleshooting

- If errors occur with JSON files, ensure they use UTF-8 encoding
- For slow performance, edit app.py to use CPU: `DEVICE = "cpu"`
- Make sure model file (best (4).pt) exists in the project directory
- Check video paths in settings match actual file locations

Is this format better? I can adjust it further to meet your needs.
