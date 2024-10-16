from enum import Enum

from au_types.action_unit import ActionUnit

class FaceEmotion(Enum):
    Happy = "Happy"
    Sad = "Sad"
    Angry = "Angry"
    Surprise = "Surprise"
    Fear = "Fear"
    Disgust = "Disgust"


class EmotionRecognizer:
    def __init__(self):
        # Mapping of emotions to required AUs
        self.emotion_au_map = {
            FaceEmotion.Happy: [ActionUnit.AU12],
            FaceEmotion.Sad: [ActionUnit.AU1, ActionUnit.AU4],
            # FaceEmotion.Angry: [ActionUnit.AU4, ActionUnit.AU7],
            FaceEmotion.Surprise: [ActionUnit.AU1, ActionUnit.AU2, ActionUnit.AU5, ActionUnit.AU26],
            FaceEmotion.Fear: [ActionUnit.AU1, ActionUnit.AU4, ActionUnit.AU5, ActionUnit.AU20],
            # FaceEmotion.Disgust: [ActionUnit.AU9, ActionUnit.AU10]
        }

    def get_emotion(self, activated_aus):
        """
        Determine the emotion based on activated AUs.

        Args:
        - activated_aus (dict): A dictionary of activated AUs with AU keys and intensity values.

        Returns:
        - FaceEmotion: The detected emotion, or None if no match is found.
        """
        if not activated_aus or not isinstance(activated_aus, dict):
            return None

        activated_au_keys = set(activated_aus.keys())

        for emotion, required_aus in self.emotion_au_map.items():
            if self._is_emotion_detected(required_aus, activated_au_keys):
                return emotion.value  # Return the matching emotion

        return None  # Return None if no emotion matches

    def _is_emotion_detected(self, required_aus, activated_au_keys):
        """
        Check if all the required AUs for an emotion are in the activated AUs.

        Args:
        - required_aus (list): A list of ActionUnit enums that must be activated for the emotion.
        - activated_au_keys (set): A set of activated AU keys.

        Returns:
        - bool: True if the emotion is detected, False otherwise.
        """
        return all(au.value in activated_au_keys for au in required_aus)
