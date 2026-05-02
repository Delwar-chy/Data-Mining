#naivebayes for spam detection
import csv
import math

# Step 1: Load dataset
data = {}
total_ham = 0
total_spam = 0

with open(r'C:\usegit\Data-Mining\dataset.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        word = row["word"].strip().lower()
        ham = int(row["ham"])
        spam = int(row["spam"])

        data[word] = {"ham": ham, "spam": spam}
        total_ham += ham
        total_spam += spam

vocab_size = len(data)

# Step 2: Classify function
def classify(text):
    words = text.lower().split()

    # Prior probabilities (log)
    log_ham = math.log(total_ham / (total_ham + total_spam))
    log_spam = math.log(total_spam / (total_ham + total_spam))

    for word in words:
        if word in data:
            log_ham += math.log((data[word]["ham"] + 1) / (total_ham + vocab_size))
            log_spam += math.log((data[word]["spam"] + 1) / (total_spam + vocab_size))
        else:
            # unseen word
            log_ham += math.log(1 / (total_ham + vocab_size))
            log_spam += math.log(1 / (total_spam + vocab_size))

    # Convert log → probability safely
    max_log = max(log_ham, log_spam)
    ham_score = math.exp(log_ham - max_log)
    spam_score = math.exp(log_spam - max_log)

    total = ham_score + spam_score
    ham_prob = ham_score / total
    spam_prob = spam_score / total

    return ham_prob, spam_prob

# Step 3: User input
if __name__ == "__main__":
    message = input("Enter a message: ")

    ham_prob, spam_prob = classify(message)

    print(f"\nHam Probability : {ham_prob:.4f}")
    print(f"Spam Probability: {spam_prob:.4f}")

    if spam_prob > ham_prob:
        print("Prediction: SPAM")
    else:
        print("Prediction: HAM")