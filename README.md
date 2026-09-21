# Image Curator

This script automatically selects a good, varied set of images from a larger image collection.

It is intended for preparing an image dataset for an identity/character LoRA. It filters unsuitable images, removes duplicates and very similar images, and tries to keep a useful variety of poses, views, and image types.

Your original image folder is not modified.

## Installation

Install Python 3, then open a Terminal or Command Prompt in the folder containing the script.

Install the required packages:

```bash
pip install -r requirements.txt
```

## Prepare your images

Put all images you want to process into one folder.

For example:

```text
my_images/
```

Subfolders are also supported.

Common formats such as JPG, PNG, WebP, BMP, TIFF, and AVIF are supported.

## Run

Use:

```bash
python main.py SOURCE OUTPUT
```

For example:

```bash
python main.py my_images curated_images
```

`my_images` is the folder containing the original images.

`curated_images` is the folder where the selected images will be written.

If you are using a custom config file:

```bash
python main.py my_images curated_images --config identity_lora.json
```

Full folder paths can also be used:

```bash
python main.py "/path/to/my/images" "/path/to/output"
```

## Output

After the script finishes, the output folder contains:

```text
curated_images/
├── train/
├── validate/
└── debug.csv
```

`train/` contains the selected training images.

`validate/` contains the validation images.

`debug.csv` contains additional information about how each image was evaluated and can be useful for troubleshooting.

## Notes

The first run may need an internet connection to download the models used for image analysis.

A compatible GPU can make processing faster, but the script can also run on the CPU.

The output folder must be different from the source image folder.
