import os
import shutil

SOURCE_FOLDER = "RAVDESS"
DEST_FOLDER = "dataset"

emotion_map = {
    "01": "neutral",
    "04": "sad",
    "05": "angry",
    "06": "fear"
}

# Create destination folders
for emotion in emotion_map.values():
    os.makedirs(os.path.join(DEST_FOLDER, emotion), exist_ok=True)

for actor_folder in os.listdir(SOURCE_FOLDER):
    actor_path = os.path.join(SOURCE_FOLDER, actor_folder)

    if os.path.isdir(actor_path):
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                emotion_code = file.split("-")[2]

                if emotion_code in emotion_map:
                    emotion_name = emotion_map[emotion_code]
                    dest_path = os.path.join(DEST_FOLDER, emotion_name)

                    shutil.copy(
                        os.path.join(actor_path, file),
                        dest_path
                    )

print("Dataset organized successfully!")
