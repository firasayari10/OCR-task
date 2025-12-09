#!/usr/bin/env python3
"""Test PHI filtering with sample prescription text."""
from main import filter_phi_with_hf

# Sample prescription text from the user
sample_text = """Name: armando Coguia Address: Went Rimbo, makati City Age: 29 Sex: M Date: 12-03-90 Hinox ) amoxicillin 500mg Cap # 21 Sig: 1 cap 3x a day for queen days. Physician's Sig. dela Cruz Lic. No. 123457 PTR No 1234567 S2 No"""

print("=" * 80)
print("Testing PHI Filter")
print("=" * 80)
print("\nOriginal text:")
print(sample_text)
print("\n" + "=" * 80)

result = filter_phi_with_hf(sample_text)

print("\n" + "=" * 80)
print("Redacted text:")
print("=" * 80)
print(result['redacted_text'])

print("\n" + "=" * 80)
print(f"PHI Summary ({len(result['phi'])} items detected):")
print("=" * 80)
for item in result['phi']:
    print(f"  [{item['label']:10s}] {item['sample']} (pos {item['start']}-{item['end']})")

print("\n" + "=" * 80)
print("Test complete!")
print("=" * 80)
