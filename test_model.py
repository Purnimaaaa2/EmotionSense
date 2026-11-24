
from transformers import pipeline

print("Loading EmotionSense AI model...")

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

print("AI model loaded successfully!")

# Test sentence
text = "I am extremely happy today!"

print("\nAnalysing text:", text)


results = classifier(text)

# Handle different Transformers output formats
if results and isinstance(results[0], list):
    results = results[0]

# Sort emotions by confidence
results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

print("\nEmotion Detection Results:")

for item in results:
    print(
        f'{item["label"].upper()}: '
        f'{item["score"] * 100:.2f}%'
    )

print("\nPredicted Emotion:", results[0]["label"].upper())