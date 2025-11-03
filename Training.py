# ner_spacy_demo.ipynb

# %% [markdown]
# ## 1. Install & Import Dependencies
# Run this only once if using Google Colab

# %%
# !pip install -U spacy scikit-learn matplotlib
# !python -m spacy download en_core_web_sm

# %%
import spacy
from spacy.training import Example
import random
import warnings
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from collections import defaultdict

warnings.filterwarnings("ignore")

# %% [markdown]
# ## 2. Load and Prepare Dataset


# %%
# Sample training data format: (text, {"entities": [(start, end, label), ...]})
TRAIN_DATA = [
    ("Apple is looking at buying U.K. startup for $1 billion", {"entities": [(0, 5, "ORG"), (27, 30, "GPE")]}),
    ("Google was founded in California", {"entities": [(0, 6, "ORG"), (20, 30, "GPE")]}),
    ("Barack Obama was born in Hawaii", {"entities": [(0, 12, "PERSON"), (26, 32, "GPE")]}),
    ("Microsoft releases Windows 11", {"entities": [(0, 9, "ORG")]}),
    ("Elon Musk leads Tesla and SpaceX", {"entities": [(0, 9, "PERSON"), (17, 22, "ORG"), (27, 33, "ORG")]}),
]

TEST_DATA = [
    ("Amazon acquired Whole Foods in Texas", {"entities": [(0, 6, "ORG"), (17, 28, "ORG"), (32, 37, "GPE")]}),
    ("Angela Merkel is from Germany", {"entities": [(0, 13, "PERSON"), (22, 29, "GPE")]}),
]

# %% [markdown]
# ## 3. Train a Custom spaCy NER Model

# %%
def train_spacy_ner(train_data, n_iter=20):
    # Start with a blank English model
    nlp = spacy.blank("en")
    
    # Create NER pipeline if not exists
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])  # ent[2] is the label

    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in train_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, losses=losses)
            print(f"Iteration {itn + 1}, Losses: {losses}")
    return nlp

# Train the model
print("Training custom NER model...")
nlp_custom = train_spacy_ner(TRAIN_DATA, n_iter=15)

# %% [markdown]
# ## 4. Evaluate Model Performance

# %%
def evaluate_model(nlp, test_data):
    true_ents = []
    pred_ents = []
    
    for text, annotations in test_data:
        doc = nlp(text)
        # Get true labels
        true_labels = ["O"] * len(doc)
        for start, end, label in annotations["entities"]:
            for i, token in enumerate(doc):
                if token.idx >= start and token.idx + len(token.text) <= end:
                    true_labels[i] = label
        
        # Get predicted labels
        pred_labels = ["O"] * len(doc)
        for ent in doc.ents:
            for i, token in enumerate(doc):
                if token in ent:
                    pred_labels[i] = ent.label_
        
        true_ents.extend(true_labels)
        pred_ents.extend(pred_labels)
    
    return true_ents, pred_ents

true_labels, pred_labels = evaluate_model(nlp_custom, TEST_DATA)

print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, zero_division=0))

# %% [markdown]
# ## 5. Visualize Results with spaCy displaCy

# %%
from spacy import displacy

sample_text = "Jeff Bezos founded Amazon in Seattle."
doc = nlp_custom(sample_text)

print(f"Input: {sample_text}")
print("Extracted Entities:")
for ent in doc.ents:
    print(f"- {ent.text} ({ent.label_})")

# Render in notebook (works in Jupyter/Colab)
displacy.render(doc, style="ent", jupyter=True)


# ## 6. Compare with Pretrained Model (en_core_web_sm)

# %%
nlp_pretrained = spacy.load("en_core_web_sm")
doc2 = nlp_pretrained(sample_text)

print("\nPretrained Model Output:")
for ent in doc2.ents:
    print(f"- {ent.text} ({ent.label_})")

displacy.render(doc2, style="ent", jupyter=True)
