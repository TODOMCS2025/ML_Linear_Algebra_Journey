"""
Day 5: Applied Analysis - Quantifying Semantic Similarity

This module focuses on practical applications of linear algebra in text analysis. It includes:
- Bag-of-Words model for text vectorization
- Cosine Similarity calculation
- Euclidean Distance calculation
- Practical solutions for numerical stability

Learning objectives:
- Understanding text vectorization using Bag-of-Words
- Calculating and interpreting Cosine Similarity
- Calculating and interpreting Euclidean Distance
- Analyzing and interpreting results in the context of text similarity

Date: 2025-08-13
"""
import numpy as np

# 1. Define the corpus
documents = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat and the dog are friends.",
    "It is sunny outside."
]

# 2. Preprocess documents (simple lowercase and split)
def preprocess(doc):
    return doc.lower().replace(".", "").split()

processed_docs = [preprocess(doc) for doc in documents]

# 3. Create vocabulary
vocab = sorted(list(set(word for doc in processed_docs for word in doc)))
vocab_index = {word: idx for idx, word in enumerate(vocab)}

# 4. Vectorize documents using Bag-of-Words
def vectorize(doc_list):
    vectors = []
    for doc in doc_list:
        vec = np.zeros(len(vocab))
        for word in doc:
            vec[vocab_index[word]] += 1
        vectors.append(vec)
    return np.array(vectors)

vectors = vectorize(processed_docs)

print("Vocabulary:", vocab)
print("Vectors:\n", vectors)

print("\n=== WHY VECTORS HAVE 14 POSITIONS? ===")
print("The vector length equals the TOTAL NUMBER OF UNIQUE WORDS across all documents")
print(f"Vector length: {len(vocab)} positions")
print(f"Total unique words in vocabulary: {len(vocab)} words")

print("\n--- COUNTING UNIQUE WORDS STEP BY STEP ---")
all_words = []
for i, doc in enumerate(documents):
    words_in_doc = preprocess(doc)
    print(f"Document {i+1}: \"{doc}\"")
    print(f"  Words: {words_in_doc}")
    all_words.extend(words_in_doc)

print(f"\nAll words (with repetitions): {all_words}")
print(f"Total word occurrences: {len(all_words)}")

unique_words = set(all_words)
print(f"\nUnique words only: {sorted(list(unique_words))}")
print(f"Number of unique words: {len(unique_words)}")

print(f"\nTherefore: Each vector needs {len(unique_words)} positions!")
print("Each position represents one unique word from the entire vocabulary")

print("\n--- BREAKDOWN BY DOCUMENT ---")
for i, doc in enumerate(documents):
    words = preprocess(doc)
    unique_in_doc = set(words)
    print(f"Document {i+1}: {len(words)} words total, {len(unique_in_doc)} unique")
    print(f"  Unique words: {sorted(list(unique_in_doc))}")

print("\n--- WHY NOT SHORTER OR LONGER? ---")
print("❌ Can't be SHORTER: We need space for every possible word")
print("❌ Can't be LONGER: We only have these 14 unique words")
print("✅ Exactly 14: One position for each unique word in our corpus")

print("\n--- WHAT IF WE ADD MORE DOCUMENTS? ---")
new_doc = "The bird flies in the sky."
new_words = preprocess(new_doc)
new_unique = set(new_words)
current_vocab = set(vocab)
additional_words = new_unique - current_vocab

print(f"If we add: \"{new_doc}\"")
print(f"New words: {new_words}")
print(f"Additional unique words: {sorted(list(additional_words))}")
print(f"New vocabulary size would be: {len(vocab)} + {len(additional_words)} = {len(vocab) + len(additional_words)}")
print("So vectors would need to grow to accommodate new words!")

print("\n=== UNDERSTANDING VECTOR REPRESENTATION ===")
print("Each vector represents a document, and each position represents a word")
print("The number at each position shows HOW MANY TIMES that word appears")

print("\n--- VOCABULARY TO INDEX MAPPING ---")
for word, idx in vocab_index.items():
    print(f"Position {idx:2d}: '{word}'")

print("\n--- DETAILED VECTOR BREAKDOWN ---")
for i, (doc, vector) in enumerate(zip(documents, vectors)):
    print(f"\nDocument {i+1}: \"{doc}\"")
    print(f"Processed: {processed_docs[i]}")
    print(f"Vector: {vector}")
    
    print("Word counts:")
    for j, count in enumerate(vector):
        if count > 0:  # Only show words that appear
            print(f"  Position {j:2d} ('{vocab[j]}'): {int(count)} times")
    
    print("Vector breakdown by position:")
    vector_breakdown = [f"{vocab[j]}:{int(count)}" for j, count in enumerate(vector) if count > 0]
    print(f"  {' | '.join(vector_breakdown)}")

print("\n=== VECTOR SIMILARITY EXPLANATION ===")
print("Why do documents 1 and 2 have high similarity (0.75)?")
print("Doc 1: 'The cat sat on the mat.'")
print("Doc 2: 'The dog sat on the log.'")
print("\nCommon words: 'sat', 'on', 'the' (appears twice in both)")
print("Different words: 'cat'/'mat' vs 'dog'/'log'")

# Let's manually show the similarity calculation
vec1, vec2 = vectors[0], vectors[1]
print(f"\nVector 1: {vec1}")
print(f"Vector 2: {vec2}")
print(f"Dot product: {np.dot(vec1, vec2)}")
print(f"Norm of vec1: {np.linalg.norm(vec1):.3f}")
print(f"Norm of vec2: {np.linalg.norm(vec2):.3f}")
print(f"Cosine similarity: {np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)):.3f}")

print("\n=== STEP-BY-STEP COSINE SIMILARITY CALCULATION ===")
print("Cosine similarity = (A·B) / (||A|| × ||B||)")
dot_prod = 0
print("Dot product calculation:")
for i in range(len(vec1)):
    if vec1[i] > 0 or vec2[i] > 0:
        contribution = vec1[i] * vec2[i]
        print(f"  Position {i:2d} ('{vocab[i]}'): {vec1[i]} × {vec2[i]} = {contribution}")
        dot_prod += contribution
print(f"Total dot product: {dot_prod}")

print(f"\nMagnitude calculations:")
print(f"||Vec1|| = √(sum of squares) = √({sum(vec1**2)}) = {np.linalg.norm(vec1):.3f}")
print(f"||Vec2|| = √(sum of squares) = √({sum(vec2**2)}) = {np.linalg.norm(vec2):.3f}")

print(f"\nFinal similarity: {dot_prod} / ({np.linalg.norm(vec1):.3f} × {np.linalg.norm(vec2):.3f}) = {dot_prod / (np.linalg.norm(vec1) * np.linalg.norm(vec2)):.3f}")

print("\n=== UNDERSTANDING THE RESULTS ===")
print("High similarity (close to 1): Documents are very similar")
print("Low similarity (close to 0): Documents are different") 
print("Similarity = 1: Documents are identical")
print("Similarity = 0: Documents have no words in common")

print("\n--- ALL PAIRWISE SIMILARITIES ---")
for i in range(len(vectors)):
    for j in range(i+1, len(vectors)):
        # Calculate cosine similarity directly
        dot_product = np.dot(vectors[i], vectors[j])
        norm_product = np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
        sim = dot_product / norm_product if norm_product != 0 else 0
        print(f"Doc {i+1} vs Doc {j+1}: {sim:.3f}")
        print(f"  \"{documents[i]}\"")
        print(f"  \"{documents[j]}\"")
        print()

# 5. Comparison metrics
def calculate_cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if norm_product == 0:
        return 0
    return dot_product / norm_product

def calculate_euclidean_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

# 6. Example comparisons
print("\nCosine similarity between doc1 and doc2:",
      calculate_cosine_similarity(vectors[0], vectors[1]))
print("Euclidean distance between doc1 and doc2:",
      calculate_euclidean_distance(vectors[0], vectors[1]))

"""
Analysis and Interpretation:

1. Most similar pair according to Cosine Similarity:
   - doc1 ("The cat sat on the mat.") and doc3 ("The cat and the dog are friends.") are likely most similar.
   - This matches human intuition because they both talk about cats (and partially about dogs), sharing more common words.

2. Most similar pair according to Euclidean Distance:
   - doc1 and doc2 ("The dog sat on the log.") may appear closer by Euclidean distance because their total word counts and sentence lengths are similar.
   - Cosine similarity focuses on direction (word distribution), while Euclidean considers absolute magnitude, so results can differ.

3. doc4 ("It is sunny outside.") has low similarity/high distance with others:
   - It shares very few or no words with the other documents.
   - This leads to low cosine similarity and high Euclidean distance.

4. Limitations of the Bag-of-Words model:
   - Word order is lost (e.g., "cat sat on the mat" vs. "mat sat on the cat").
   - Context and meaning are lost (e.g., "dog" and "puppy" are treated as unrelated).
   - Synonyms, negations, and semantic relationships are ignored.
"""

print("=" * 50)
print("✅ DAY 5 COMPLETE!")
print("📚 Key Learnings:")
print("   • Text vectorization using Bag-of-Words")
print("   • Cosine Similarity calculation")
print("   • Euclidean Distance calculation")
print("   • Analyzing and interpreting results in the context of text similarity")
print("=" * 50)