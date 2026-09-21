# DC-808

DC-808 is a tool for automatically selecting a good, varied set of images from a larger character image collection.

Unlike simple filtering or random selection, it looks at the dataset as a whole. It removes unsuitable and duplicate images while trying to preserve useful variety across poses, views, framing, expressions, and other visual characteristics.

### The idea

Audit the images → remove duplicates → compare the dataset → select a balanced subset

DC-808 is designed to reduce the manual work involved in preparing character LoRA datasets while keeping the resulting image set varied and easy to review.

## Usage

```bash
python main.py SOURCE OUTPUT
```

For example:

```bash
python main.py my_images curated_images
```

A custom configuration file can also be supplied:

```bash
python main.py my_images curated_images --config identity_lora.json
```

The original source folder is not modified.

Selected images are written to separate training and validation folders:

```text
curated_images/
├── train/
├── validate/
└── debug.csv
```

`debug.csv` contains additional information about how the images were evaluated and selected.

# Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

# License

This project is licensed under the MIT License.

# Acknowledgements

Martin Bosgra: Author and primary maintainer of the project.
