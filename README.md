# Complete Tutorial: How to Run the Parking Management System


## Step 1: Clone the Repository

1. Open your terminal or command prompt
2. Clone the repository by running:
   ```bash
   git clone https://github.com/lialavy/lia_lavy_parking_yolo.git
   ```
3. Navigate to the project directory:
   ```bash
   cd lia_lavy_parking_yolo
   ```

### About .gitattributes

The repository includes a `.gitattributes` file which ensures consistent handling of text and binary files across different operating systems. This is particularly important for:

- JSON files (like bounding_boxes.json)
- Video files (like carPark.mp4)
- Model files (like best (4).pt)

This file automatically configures Git to handle file encodings properly, so you don't need to worry about encoding issues when cloning the repository. The system is set up to work consistently across Windows, macOS, and Linux systems.

## Step 2: Set Up the Environment

1. Create a virtual environment (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This will install all the necessary packages including Flask, OpenCV, PyTorch, and Ultralytics YOLO.

## Step 3: Run the Application

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. The server will start running. You should see output like:
   ```
   * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
   ```

3. Open a web browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Step 4: Create an Account and Log In

1. When you first access the application, you'll be redirected to the login page
2. Click on "Sign up" to create a new account
3. Fill in the registration form:
   - Choose a username
   - Enter your email address (if you want admin access, change: lia.ninio24@gmail.com to your email in the app.py code)
   - Create a password
4. Click "Sign Up" to create your account
5. Log in with your new credentials

## Step 5: Using the Parking Management System

After logging in, you'll see the main dashboard with several options:

### View the Dashboard
- Click on "View Dashboard" to see the real-time parking detection
- The system will display:
  - Number of occupied parking spots
  - Number of available parking spots
  - Live video feed with parking spaces highlighted
  - Green boxes indicate available spaces
  - Red boxes indicate occupied spaces

### View History
- Click on "View History" to see a log of parking space changes over time
- This shows when parking spaces became occupied or available

### Admin Features (only if using the admin email)
- Click on "Admin Dashboard" in the navigation bar
- Here you can:
  - Add new parking lots
  - Edit existing parking lot configurations
  - Delete parking lots
  - Set video sources and model paths

## Step 6: Creating Your Own Parking Spaces

If you want to set up the system for a new parking area:

1. From the admin dashboard, click "Add New Parking Lot"

2. Provide details:
   - **Lot ID**: A unique identifier (e.g., "my_lot")
   - **Lot Name**: A display name (e.g., "My Parking Lot")
   
   - **Video Source**: 
     * For included videos: Use just the filename (e.g., "carPark.mp4")
     * For your own videos: Use the full path if not in the root directory (e.g., "C:/Videos/my_parking.mp4")
     * For webcam: Use "0" for the default camera

   - **Model Path**: 
     * For the included model: Use "best (4).pt"
     * For your own models: Use the full absolute path (e.g., "C:/Models/yolov8n.pt")

   - **Bounding Boxes File**: 
     * For existing files: Use just the filename (e.g., "bounding_boxes.json")
     * For new files: Use the full path if not in the root directory (e.g., "C:/Configs/my_lot_boxes.json")

3. Important notes about paths:
   - The system expects either:
     * Files in the root directory of the project (use just the filename)
     * Files elsewhere on your system (use the full absolute path)
   - On Windows, use forward slashes in paths (/) even though backslashes (\) are typical
   - Avoid using spaces in filenames when possible, or enclose paths with spaces in quotes

4. To create parking space bounding boxes:
   - Use the ParkingPtsSelection tool which will let you:
     * Upload a snapshot of your parking area
     * Click on four corners of each parking space (in clockwise or counter-clockwise order)
     * Save the resulting JSON file for use with your parking lot

5. After creating a new parking lot:
   - It will appear in the dropdown menu on the dashboard
   - Select it to start detecting parking spaces in that area
   - The system will use the specified model to detect vehicles
   - Detected vehicles will be compared against your defined parking spaces
   - Spaces with a vehicle inside will be marked as occupied

## Step 7: Understanding the System Components

### Key Components:

1. **app.py**: The main Flask application that handles web routes, user authentication, and the web interface
2. **parking_management.py**: Contains the core logic for processing video frames and detecting parking spaces
3. **bounding_boxes.json**: Defines the coordinates of parking spaces
4. **best (4).pt**: The YOLO model file used for vehicle detection
5. **users.json**: Contains user account information
6. **parking_lots.json**: Stores configurations for different parking areas

### How Detection Works:

1. The system uses YOLO (You Only Look Once) to detect vehicles in video frames
2. It compares the positions of detected vehicles with the predefined parking space coordinates
3. If a vehicle is detected within a parking space, it's marked as occupied
4. The system updates the dashboard with real-time counts of available and occupied spaces

## Troubleshooting

If you encounter any issues:

1. **Encoding errors with JSON files**: 
   - The repository includes a `.gitattributes` file that should prevent encoding issues
   - If you still get encoding errors, open the JSON file in a text editor and re-save it with UTF-8 encoding

2. **Git-related issues**:
   - If you encounter strange file encoding issues or line ending problems, the `.gitattributes` file in the repository should handle this
   - If you still have problems, you can run:
     ```bash
     git config --global core.autocrlf false
     git clone https://github.com/lialavy/lia_lavy_parking_yolo.git
     ```

3. **Model loading errors**:
   - Make sure the YOLO model file (best (4).pt) exists in the root directory
   - If needed, download a compatible YOLOv5 or YOLOv8 model and update the path in the settings
   - If you get CUDA errors, try setting the device to "cpu" in app.py: `DEVICE = "cpu"`

4. **Video source not found**:
   - Check that the video files referenced in the parking_lots.json file exist in the correct location
   - For webcam access, try changing the video source to "0" (without quotes)
   - Make sure video files are properly formatted (MP4, AVI, etc.)

5. **Dependencies issues**:
   - If you encounter missing dependency errors, try installing them individually:
     ```bash
     pip install ultralytics opencv-python flask numpy torch
     ```
   - For GPU support, make sure you have the correct CUDA version installed for your PyTorch version

6. **Application crashes or freezes**:
   - Check the terminal for error messages
   - Ensure your system meets the minimum requirements for running YOLO models
   - Try reducing the resolution of your video source if processing is slow

## Additional Tips

- The system works best with a clear overhead view of parking spaces
- For best performance, use a GPU-enabled system (though CPU is supported)
- You can customize the detection parameters in the ParkingManagement class in parking_management.py
- Sample videos are included in the "videos and screenshots" folder for testing
- The `.gitattributes` file ensures consistent file handling across different operating systems
- For large parking lots, consider dividing them into multiple smaller areas for better performance
- If performance is slow, try using a lighter YOLO model (like YOLOv8n instead of larger variants)

## Extending the System

If you want to extend or customize the system:

1. **Adding new features**:
   - The Flask application structure makes it easy to add new routes and views
   - Check app.py for examples of existing routes

2. **Customizing the detection**:
   - The ParkingManagement class in parking_management.py can be modified to adjust detection parameters
   - You can change the colors used for visualization by modifying the color variables (arc, occ, dc)

3. **Integrating with other systems**:
   - The /status endpoint returns JSON data that can be consumed by other applications
   - You can build additional integrations using this API

That's it! You should now have a working parking management system that detects available and occupied parking spaces in real-time.
