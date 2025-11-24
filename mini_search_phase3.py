"""
Mini Search Engine - Phase 3 (Optimized)
Author: Alexandria Roberts
Course: MSCS 532 - Data Structures and Algorithms
Date: November 2025

Phase 2: Proof-of-concept mini search engine using an inverted i
Phase 3: Optimized and scaled version with performance comparison.
ndex and TF-IDF.

This file:
- Keeps the original small demo from Phase 2 (5 docs + test queries)
- Adds Phase 3 optimizations and larger-scale performance testing
"""

import math
import time
import heapq
from collections import defaultdict, Counter


class MiniSearchEngine:
    """
    Inverted-index search engine with TF-IDF ranking.

    Phase 3 optimizations: 
    - Use sets for posting (no repeated sort/dedup on every insert).
    - Cache IDF values to avoid recomputing logs.
    - Use heapq.nlargest for top-k ranking instead of sorting the full list.
    """

    def __init__(self):
        self.docs = {}
        self.postings = defaultdict(set)
        self.tf = Counter()
        self.df = Counter()
        self.N = 0
        self.idf_cache = {}

    def _tokenize(self, text: str):
        """Normalize and split into lowercase tokens"""
        clean = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in text)
        return [t for t in clean.split() if t]

    def add_doc(self, doc_id: int, text: str):
        """Add a new document (id, text) into index.
        Phase 2: built TF, DF, and postings.
        Phase 3: posting now use a set so we do not sort/dedup on each insert.
        """
        self.docs[doc_id] = text
        self.N = len(self.docs)

        tokens = self._tokenize(text)
        counts = Counter(tokens)

        for term, c in counts.items():
            self.tf[(term, doc_id)] = c
            self.df[term] += 1
            self.postings[term].add(doc_id)


        self.idf_cache.clear()


    def _idf(self, term: str) -> float:
        if term not in self.idf_cache:
            self.idf_cache[term] = math.log(1 + (self.N / (1 + self.df[term])))
        return self.idf_cache[term]

    def _tfidf(self, term: str, doc_id: int) -> float:
        tf = self.tf[(term, doc_id)]
        idf = self._idf(term)
        return (1 + math.log(tf)) * idf
    
    def search(self, query: str, k: int = 5):
        """Return top-k docs ranked by TF-IDF score.
        Phase 2: sorted full list of scores.
        Phase 3: uses heapq.nlargest for more efficient top-k selection.
        """
        terms = self._tokenize(query)
        if not terms:
            return [] #Handle empty or an invalid query
        
        candidates = set()
        for t in terms:
            candidates.update(self.postings.get(t, []))

        scores = []
        for d in candidates:
            s = sum(self._tfidf(t, d) for t in terms if (t, d) in self.tf)
            scores.append((s, d))
       
       # More efficient top-k: O( n log k) instead of sorting all scores 
        return heapq.nlargest(k, scores)
    

# -------------------------------
# Helper functions for Phase 3
# -------------------------------

def build_engine(engine, docs: dict):
    """Add all documents to given engine and return build times (seconds)."""
    start = time.perf_counter()
    for doc_id, text in docs.items():
        engine.add_doc(doc_id, text)
    end = time.perf_counter()
    return end - start

def time_queries(engine, queries):
    """Run all queries on the engine and return average time per query (seconds)."""
    total_time = 0.0
    for q in queries:
        start = time.perf_counter()
        _ = engine.search(q)
        end = time.perf_counter()
        total_time += (end - start)
    return total_time / max(len(queries), 1)

def make_small_docs():
    """Phase 2: original small 5-document dataset used for correctness."""
    return {
        0: "Cats like pillows",
        1: "Dogs like couches",
        2: "Cats and dogs like treats",
        3: "Milk is good for cats",
        4: "Bones are good for dogs"
    }

def make_large_docs(n: int):
    """
    Phase 3: generate a simple synthetic dataset with n documents to test scaling and performance.
    """
    
    docs = {}
    for i in range(n):
        if i % 2 == 0:
            docs[i] = f"Cats like pillows and treats {i}"
        else:
            docs[i] = f"Dogs like couches and bones {i}"
    return docs

# -------------------------------------------------------
# MAIN: Phase 2 demo + Phase 3 performance/scaling tests
# -------------------------------------------------------


if __name__ == "__main__":

    print("=== Phase 2 Demo: Correctness on Small Dataset (5 docs) ===")
    engine = MiniSearchEngine()
    small_docs = make_small_docs()


    # Add documents to engine
    for doc_id, text in small_docs.items():
        engine.add_doc(doc_id, text)

    # Regular and edge-case queries 

    queries = [
        "cats", # normal single-term
        "dogs", # normal single-term
        "cats milk", # multi-term query
        "bird", # no matches (edge case)
        "CATS!", # punctuation + uppercase (edge case)
          "" # empty query (edge case)

          ]
    
    for q in queries:
        start = time.perf_counter()
        results = engine.search(q)
        end = time.perf_counter()

        print(f"\nQuery: {repr(q)}")

        if not results:
            print(" No matching documents found.")
        else:
            for score, doc_id in results:
                print(f" doc {doc_id}: {engine.docs[doc_id]} (score={score:.3f})")

        print(f"Time: {(end - start) * 1000:.3f} ms")

        # --------------------------------------------
        # Phase 3: Scaling and performance comparison
        # --------------------------------------------

    print("\n\n===Phase 3 Demo: Scaling and Performance ===")
            
    large_sizes = [1000, 5000]

    perf_queries = ["cats", "dogs", "cats treats"]

    for size in large_sizes:
            print(f"\n--- Dataset size: {size} documents ---")
            large_docs = make_large_docs(size)

            engine_large = MiniSearchEngine()

            build_time = build_engine(engine_large, large_docs)

            avg_query_time = time_queries(engine_large, perf_queries)

            print(f"Index build time: {build_time:.4f} secoonds")
            print(f"Average query time: {avg_query_time * 1000:.4f} ms over {len(perf_queries)} queries")


