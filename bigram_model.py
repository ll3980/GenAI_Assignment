"""The Module 2 bigram generator, organized for the Module 3 API."""

from collections import Counter, defaultdict
import random
import re


class BigramModel:
    """Learn next-word probabilities from a small collection of texts."""

    def __init__(self, corpus: list[str]):
        self.next_word_counts: dict[str, Counter] = defaultdict(Counter)
        self.vocabulary: set[str] = set()

        # Process each text separately so unrelated texts are not joined.
        for text in corpus:
            words = re.findall(r"\b\w+\b", text.lower())
            self.vocabulary.update(words)
            for current_word, next_word in zip(words, words[1:]):
                self.next_word_counts[current_word][next_word] += 1

        # Only words with an observed successor belong in the denominator.
        # Each nonempty probability row therefore sums to one.
        self.bigram_probs = {
            word: {
                next_word: count / sum(counts.values())
                for next_word, count in counts.items()
            }
            for word, counts in self.next_word_counts.items()
        }

    def generate_text(self, start_word: str, length: int = 20) -> str:
        """Sample up to `length` words, including the starting word."""
        start_word = start_word.strip().lower()
        if not start_word or not start_word.isalpha():
            raise ValueError("start_word must be one alphabetic word.")
        if not 1 <= length <= 200:
            raise ValueError("length must be between 1 and 200.")
        if start_word not in self.vocabulary:
            raise KeyError(start_word)

        generated_words = [start_word]
        current_word = start_word
        for _ in range(length - 1):
            probabilities = self.bigram_probs.get(current_word)
            if not probabilities:
                # Stop when the corpus provides no next-word observation.
                break
            current_word = random.choices(
                list(probabilities), weights=list(probabilities.values()), k=1
            )[0]
            generated_words.append(current_word)
        return " ".join(generated_words)
