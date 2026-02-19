import re
import numpy as np

def extract_features(text):
    
    words = text.split()
    lines = text.split('\n')
    
    word_count = len(words)
    char_count = len(text)
    
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    digit_count = sum(c.isdigit() for c in text)
    
    uppercase_count = sum(c.isupper() for c in text)
    uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
    
    num_lines = len(lines)
    
    return np.array([
        word_count,
        char_count,
        avg_word_length,
        digit_count,
        uppercase_ratio,
        num_lines
    ])
