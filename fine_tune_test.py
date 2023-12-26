from transformers import pipeline

text = "Da li biste bili ljubazni da mi otvorite vrata, dragi moj Nikola?"
classifier = pipeline("token-classification", model="C:\\Users\\mihailo\\OneDrive - Faculty of Mining and Geology\\resursi\\bert modeli\\bertovic-base-pos")

for x in classifier(text):
    print("\t".join([x["word"], x["entity"]]))
