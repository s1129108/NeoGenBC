#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("-query", "--query_path", type=str, required=True, help="Path to the query .npy file")
parser.add_argument("-db", "--database_path", type=str, required=True, help="Path to the RAG database .npy file")
parser.add_argument("-out", "--output_path", type=str, required=True, help="Path to save the fused embeddings")
parser.add_argument("-batch", "--batch_size", type=int, default=100, help="Number of query sequences per batch")
# parser.add_argument("-ratio","--query_ratio", type=float, default=0.7, help="Query ratio for fusing")
args = parser.parse_args()

# Parameters
query_path = args.query_path
database_path = args.database_path
output_path = args.output_path
batch_size = args.batch_size
# query_ratio = args.query_ratio
# Fixed parameters
maxseq = 35
num_feature = 1024  # Embedding dimension

# Load Query and RAG Database
query_data = np.load(query_path)  # Shape: (N, 1, 35, 1024)
rag_database = np.load(database_path)  # Shape: (N, 1, 35, 1024)

# Function to find the top 5 most similar sequences and fuse embeddings
def compute_rag_embedding(query, database, maxseq, num_feature):
    """
    Finds the 5 closest sequences in the database, averages them, and fuses the result with the query embedding.
    """
    query = query.reshape(1, maxseq, num_feature)  # Reshape for consistency
    database = database.reshape(database.shape[0], maxseq, num_feature)  # Remove singleton dimension

    # Compute L2 distances efficiently
    distances = np.linalg.norm(database - query, axis=(1, 2))  # Compute distances across all sequences

    # Get indices of 5 closest sequences
    top_5_indices = np.argsort(distances)[:5]
    closest_sequences = database[top_5_indices]  # Retrieve the closest 5 embeddings

    # Compute the average of top-5 sequences
    avg_embedding = np.mean(closest_sequences, axis=0)

    # Fuse query and RAG embedding (90% query, 10% RAG)
    # 
    fused_embedding = (query * 0.5) + (avg_embedding * 0.5)

    return fused_embedding.reshape(1, maxseq, num_feature)  # Reshape to original format

# Initialize an array to store all fused embeddings (same as query_data)
fused_embeddings = np.copy(query_data)

# Process Query Data in Batches
num_queries = query_data.shape[0]
num_batches = (num_queries + batch_size - 1) // batch_size

print(f"Total queries: {num_queries}")
# print(f"Fusing with ratio: Query {query_ratio} and RAG {(1-query_ratio):.1f}")
print("Fusing with ratio: Query 0.9 and RAG 0.1")
print(f"Processing in {num_batches} batches of size {batch_size}")

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_queries)
    batch_queries = query_data[start_idx:end_idx]

    for i, query in enumerate(batch_queries):
        fused_embedding = compute_rag_embedding(query, rag_database, maxseq, num_feature)
        fused_embeddings[start_idx + i] = fused_embedding  # Update all samples

    print(f"Batch {batch_idx + 1}/{num_batches} processed")

# Save the final fused embeddings
np.save(output_path, fused_embeddings)
print(f"Fused embeddings saved to {output_path}")