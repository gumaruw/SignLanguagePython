# SignFlow | Real-Time Sign Language Alphabet Recognition System

## Overview
This project is a real-time system for recognizing English Sign Language letters. Developed at Doğuş University by the Software Engineering team, it uses machine learning and computer vision to help hearing-impaired individuals communicate more easily and support sign language learning.

The system recognizes letters with over 90% accuracy and handles dynamic gestures like "J" and "Z". It is designed to be simple, reliable, and accessible to users of all ages.

## Project Structure
SignFlow/  
├── Data/                   - Dataset  
├── gui.py                  - GUI application  
├── egitim.py               - Training script  
├── import os.py            - Helper script  
├── test.py                 - Test script  
├── test_model.py           - Model testing  
├── train_model.py          - Model training  
├── requirements.txt           
├── README.md                   
├── .gitignore               


## Key Features
- **Real-Time Recognition:** Detects hand gestures instantly using a camera.  
- **High Accuracy:** Recognition accuracy exceeds 90%.  
- **Dynamic Gesture Support:** Handles complex gestures like "J" and "Z".  
- **User-Friendly:** Simple interface suitable for everyone.  
- **Inclusive Impact:** Helps raise awareness of sign language and accessibility.

## Goals
- Enable hearing-impaired individuals to communicate more easily.  
- Provide an educational tool for learning sign language.  
- Promote societal awareness and inclusivity.

## Technical Details
- **Framework:** TensorFlow  
- **Language:** Python  
- **Libraries:** OpenCV, Numpy  
- **Model:** Convolutional Neural Network (CNN)  
- **Dataset:** Custom dataset with static and dynamic gestures

## How It Works
1. **Input:** Capture hand gestures via camera.  
2. **Preprocessing:** Resize, normalize, and prepare images for recognition.  
3. **Prediction:** CNN processes images and identifies letters.  
4. **Output:** Display letters in real-time, optionally with voice feedback.

## Installation
```bash
git clone https://github.com/YourUsername/SignFlow.git
cd SignFlow
pip install -r requirements.txt
python main.py
```

## Performance
- Accuracy: >90% on training and validation datasets.
- Dynamic Gestures: Successfully recognizes "J" and "Z".
- Low-End Device Support: Runs efficiently on basic hardware.
- User Feedback: Simple, accurate, and practical.

## Contributors
- Ayşe Ceren Doğan
- Cemre Dağ
- Emir Ekrem Kaya
- Hatice Uçar

## Note
The project name is intended for future research references and was not used during the original research project.
