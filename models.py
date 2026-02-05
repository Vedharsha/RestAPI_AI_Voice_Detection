from transformers import pipeline
import librosa
import numpy as np

# Load model once at startup (important for performance)
classifier = pipeline(
    "audio-classification",
    model="Hemgg/Deepfake-audio-detection",
    device=-1  # CPU (use 0 if GPU available)
)


def detect_audio(y: np.ndarray) -> tuple[str, float, str]:
    """
    Detect if audio is AI_GENERATED or HUMAN.

    Returns:
    - classification (str)
    - confidenceScore (float 0-1)
    - explanation (str)
    """

    try:
        # Send audio correctly to HuggingFace
        result = classifier({
            "array": y,
            "sampling_rate": 16000
        })

        # Safety check
        if not result or len(result) == 0:
            return "HUMAN", 0.50, "Insufficient audio features detected."

        # Get top prediction
        top = result[0]
        label = top["label"].lower()
        score = float(top["score"])

        # Label mapping
        if any(word in label for word in ["ai", "fake", "synthetic", "deepfake"]):
            classification = "AI_GENERATED"
        else:
            classification = "HUMAN"

        confidence = round(score, 3)

        # -------------------------
        # Feature-based analysis
        # -------------------------

        # Spectral flatness (robotic vs natural)
        flatness = float(librosa.feature.spectral_flatness(y=y).mean())

        # Pitch variation
        pitch = librosa.yin(y, fmin=75, fmax=300)
        pitch = pitch[~np.isnan(pitch)]  # remove NaN values

        pitch_std = float(np.std(pitch)) if len(pitch) > 0 else 0.0

        cues = []

        if flatness > 0.5:
            cues.append("unnatural spectral flatness (robotic tone)")
        else:
            cues.append("natural spectral variation")

        if pitch_std < 10:
            cues.append("low pitch variation (synthetic pattern)")
        else:
            cues.append("normal pitch variation")

        # Feature vote
        feature_vote = (
            "AI_GENERATED"
            if flatness > 0.5 and pitch_std < 10
            else "HUMAN"
        )

        cues_text = " and ".join(cues)

        # Explanation logic
        if feature_vote == classification:
            explanation = (
                f"{cues_text}, supporting the model prediction "
                f"of {classification.lower()} voice."
            )
        else:
            explanation = (
                f"{cues_text}. However, the deep learning model "
                f"detected patterns of {classification.lower()} voice."
            )

        explanation = explanation.capitalize()

        return classification, confidence, explanation

    except Exception as e:

        # Fallback protection
        return (
            "HUMAN",
            0.50,
            f"Audio analysis failed: {str(e)}. Defaulted to human voice."
        )
