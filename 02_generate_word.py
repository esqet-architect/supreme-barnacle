#!/usr/bin/env python3
"""
Part 2: Generate Tribonacci word (no graphics, just the sequence)
This uses string manipulation, runs quickly even on phone.
"""

def generate_tribonacci_word(n_steps=5000):
    subs = {'1': '12', '2': '13', '3': '1'}
    word = ['1']
    while len(''.join(word)) < n_steps:
        word = [c for w in word for c in subs[w]]
    return ''.join(word)[:n_steps]

print("="*50)
print("GENERATING TRIBONACCI WORD")
print("="*50)

for length in [1000, 2000, 5000]:
    word = generate_tribonacci_word(length)
    freq1 = word.count('1') / len(word)
    freq2 = word.count('2') / len(word)
    freq3 = word.count('3') / len(word)
    print(f"\nLength {length}:")
    print(f"  '1': {freq1:.4f}  '2': {freq2:.4f}  '3': {freq3:.4f}")
    print(f"  Ratio expected β²:β:1 ≈ {freq1/freq3:.4f}:{freq2/freq3:.4f}:1")

# Save word for later
word = generate_tribonacci_word(8000)
with open('data/tribonacci_word.txt', 'w') as f:
    f.write(word)
print("\n✅ Word saved to data/tribonacci_word.txt")
