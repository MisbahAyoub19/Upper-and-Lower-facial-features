import os
import pandas as pd

annotation_path = "D:/Pycharm_result/Emotion/7Mar/annotation/oulu-vi"
videos_path = "D:/Pycharm_result/Emotion/OULU/VI"
emotion_mapping = {'A': 0, 'D': 1, 'F': 2, 'H': 3, 'S1': 4, 'S2': 5}

#ADFES Annotations
#emotion_mapping c = {'Anger': 0, 'Contempt': 1, 'Disgust': 2,  'Embarrass': 3, 'Fear': 4, 'Joy': 5, 'Neutral': 6, 'Pride': 7, 'Sadness': 8, 'Surprise': 9}


if not os.path.exists(annotation_path):
    os.makedirs(annotation_path)

def get_emotion(video_name):
    # Extract emotion from the last word of the video name
    words = video_name.split('_')
    last_word = words[-1].split('.')[0]
    return emotion_mapping.get(last_word, None)

def generateAnnotations(num):
    videos = os.listdir(videos_path)
    for video in videos:
        video_name, extension = os.path.splitext(video)
        emotion = get_emotion(video_name)

        if emotion is not None:
            # Create DataFrame with numeric emotion label
            annotation_df = pd.DataFrame(columns=["emotion"])

            for i in range(num):
                frame_id = f"{video_name}_{i}"  # Create frame ID using video name and index
                annotation_df = annotation_df.append({'emotion': emotion}, ignore_index=True)

            annotation_df.to_csv(os.path.join(annotation_path, video_name + '.csv'), index=False,
                                 columns=["emotion"])
        else:
            print(f"Emotion not found in '{video_name}', skipping.")

if __name__ == '__main__':
    generateAnnotations(50)