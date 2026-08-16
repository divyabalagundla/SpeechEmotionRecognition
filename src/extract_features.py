import os
import librosa
import numpy as np
import pandas as pd
DATASET_PATH = r"C:\Users\divya\OneDrive\Desktop\SpeechEmotionRecognition\dataset"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

features = []
labels = []

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for filename in os.listdir(actor_path):

        if not filename.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, filename)

        parts = filename.split("-")
        emotion_code = parts[2]

        emotion = emotion_map.get(emotion_code)

        if emotion is None:
            continue

        try:
            audio, sample_rate = librosa.load(file_path, sr=16000)

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=40
            )

            mfcc_mean = np.mean(mfcc, axis=1)

            features.append(mfcc_mean)
            labels.append(emotion)

        except Exception as e:
            print("Error:", file_path)
            print(e)

df = pd.DataFrame(features)
df["emotion"] = labels

os.makedirs("results", exist_ok=True)

df.to_csv("results/emotion_features.csv", index=False)

print("\nFeature extraction completed!")
print("Total audio files processed:", len(df))
print("\nEmotion distribution:")
print(df["emotion"].value_counts())