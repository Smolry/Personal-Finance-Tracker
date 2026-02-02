from src.ml_categorizer import MLCategorizer

# Load latest model
categorizer = MLCategorizer()
categorizer.load_model()

# Test predictions
test_desc = ["Starbucks Coffee", "Walmart Groceries", "Netflix"]
predictions = categorizer.predict(test_desc)
confidences = categorizer.get_confidence(test_desc)

for desc, pred, conf in zip(test_desc, predictions, confidences):
    print(f"{desc} → {pred} ({conf:.2%} confidence)")