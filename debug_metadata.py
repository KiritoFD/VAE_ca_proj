import json

metadata_path = "./wikiart_dataset/metadata.json"

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print("=" * 50)
print("Metadata Structure:")
print("=" * 50)
print(f"Keys: {metadata.keys()}")
print(f"\nTotal samples: {metadata['total_samples']}")
print(f"Number of styles: {metadata['num_styles']}")
print(f"\nValid styles type: {type(metadata['valid_styles'])}")
print(f"Valid styles length: {len(metadata['valid_styles'])}")
print(f"\nFirst 5 styles:")
for i, style in enumerate(metadata['valid_styles'][:5]):
    print(f"  [{i}] {style!r} (type: {type(style).__name__})")

print(f"\nStyle to ID mapping (first 5):")
for i, (k, v) in enumerate(list(metadata['style_to_id'].items())[:5]):
    print(f"  {k!r} -> {v} (key type: {type(k).__name__})")

print(f"\nFirst sample:")
print(f"  {metadata['valid_samples'][0]}")
