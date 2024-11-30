import openai
from ai_scientist.perform_review import load_paper, perform_review

client = openai.OpenAI()
model = "gpt-4o-2024-05-13"

# Load paper from pdf file (raw text)
paper_txt = load_paper("successful_automata_experiment.pdf")
# Get the review dict of the review
review = perform_review(
    paper_txt,
    model,
    client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)

# Inspect review results
print("Overall:\n")
print(review["Overall"])  # overall score 1-10
print("\n")
print("Decision:\n")
print(review["Decision"])  # ['Accept', 'Reject']
print("\n")
print("Weaknesses:\n")
print(review["Weaknesses"])  # List of weaknesses (str)