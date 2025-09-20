# Andromr

Optical Music Recognition (OMR) for Android using [homr](https://github.com/liebharc/homr).

## Technical Details
Andromr uses **Kivy** and **KivyMD** for the UI, with **Buildozer** to generate the APK. The OMR engine is based on [homr](https://github.com/liebharc/homr), which provides great quality due to the use Vision Transformers. Big thanks to [Christian Liebhardt](https://github.com/liebharc) for creating and open-sourcing such a great project.

homr itself uses 2 differnent models: A segmentation model and a transformer model (encoder and decoder).
Due to performance reasons I decided to run the transformer models using onnxruntime and the CNN models using LiteRT.
Furthermore I decided to split the encoder into the CNN-backbone (running in LiteRT) and the ViT (running in onnx).

The model inference is using native Java APIs accessed via pyjnius.

## Build
*Note: Buildozer is really fragile and support by me is limited.*

Andromr is built with buildozer under WSL. If you are new to buildozer you can follow this [tutorial](https://www.youtube.com/watch?v=pzsvN3fuBA0) for setting buildozer up.

You will also have to put the recipe located in .recipes to `.buildozer\android\platform\python-for-android\pythonforandroid\recipes` and add opencv.

#### Setting up OpenCV:
1.  Download the `opencv-android-sdk` (version 4.7.0) from [opencv.org](https://opencv.org/)
3.  Copy the `native` folder from the SDK into a new folder named `opencv`, placed at the same level as your `buildozer.spec` file.

## Contributing
PRs are welcome :)

If you only want to change something about the homr backend please first think about contributing to homr. Sooner or later, all the changes made to homr will be added to Andromr too.

## Acknowledgments
Thanks to [Christian Liebhardt](https://github.com/liebharc) for open-sourcing homr.
All the other open-source licenses are listed in `data/licenses.txt`.

## Future Plans
Late 2025: LiteRT Next integration (2x performance improvement for CNNs)

Early 2026: Release of Andromr in PlayStore

Mid 2026: NPU support for LiteRT Next

## License
Andromr is open-sourced under the AGPL 3.0 license.
