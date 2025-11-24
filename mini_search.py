
"""
Mini Search Engine
Author:Alexandria Roberts
Course: Data Structures & Algorithm
Date: November 2025

Implements a minimal keyword search engine using provided indexes to demonstrate the real world data structure design and performance analysis capabilities.
"""

import math
import time
from collections import defaultdict, Counter

class MiniSearchEngine:
    """Inverted-index search engine with TF-IDF ranking."""

    def __init__(self):
        self.docs = {}
        self.postings = defaultdict(list)
        self.tf = Counter()
        self.df = Counter()
        self.N = 0

    def _tokenize(self, text: str):
        """Normalize and split into lowercase tokens"""
        clean = ''.join(ch.lower() if ch.isalnum() else ' ' for ch in text)
        return [t for t in clean.split() if t]

    def add_doc(self, doc_id: int, text: str):
        """Add a new document (id, text) into index."""
        self.docs[doc_id] = text
        self.N = len(self.docs)

        tokens = self._tokenize(text)
        counts = Counter(tokens)

        for term, c in counts.items():
            self.tf[(term, doc_id)] = c
            self.df[term] += 1
            self.postings[term].append(doc_id)
            self.postings[term] = sorted(set(self.postings[term])) 

    def _tfidf(self, term: str, doc_id: int) -> float:
        """Computing TF-IDF weight for a term in a document."""
        tf = self.tf[(term, doc_id)]
        idf = math.log(1 + (self.N / (1 + self.df[term])))
        return (1 + math.log(tf)) * idf
    
    def search(self, query: str, k: int = 5):
        """Return top-k docs ranked by TF-IDF score."""
        terms = self._tokenize(query)
        if not terms:
            return[] #Handle empty or an invalid query
        
        candidates = set()
        for t in terms:
            candidates.update(self.postings.get(t, []))

        scores = []
        for d in candidates:
             s = sum(self._tfidf(t, d) for t in terms if (t, d) in self.tf)
             scores.append((s, d))
       
        scores.sort(reverse=True)
        return scores[:k]


#------demo section (Phase 2 Proof of Concept including with Edge Cases)------

if __name__ == "__main__":
    engine = MiniSearchEngine()
    docs = {
        0: "Cats like pillows",
        1: "Dogs like couches",
        2: "Cats and dogs like treats",
        3: "Milk is good for cats",
        4: "Bones are good for dogs"
    }

    # Add documents to engine
    for doc_id, text in docs.items():
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

