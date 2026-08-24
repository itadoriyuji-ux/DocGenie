from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentence1 = "The cat sat on the mat"
sentence2 = "A kitten was resting on the rug"
sentence3 = "The stock market crashed today"

embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
embedding3 = model.encode(sentence3)

similarity_1_2 = model.similarity(embedding1, embedding2)
similarity_1_3 = model.similarity(embedding1, embedding3)

print(f"The similarity between embedding 1 and 2 is {similarity_1_2}")
print(f"The similarity between embedding 1 and 3 is {similarity_1_3}")
