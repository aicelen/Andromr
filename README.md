# Andromr

**Andromr is currently in Play Store closed testing, to release Andromr I need 12 testers. It would be awesome if you could help me release Andromr by becoming a tester. To do that you need to join this
google group and follow the instructions their. Thank you!**

Optical Music Recognition (convert a picture of music notes to a machine readable format like .musicxml) for Android using [homr](https://github.com/liebharc/homr).

## How to use
- Take a picture inside the app (and take more if you want)
- Run optical music recognition (leave the app focused)
- Export via the Share button

## Features
- All features of homr (near state-of-the-art optical music recognition)
- Multi Page
- Privacy (your scores stay on your device)

## Technical Details
Andromr uses **Kivy** and **KivyMD** for the UI, with **Buildozer** to generate the APK. The OMR engine is based on [homr](https://github.com/liebharc/homr), which provides great quality due to the use of Vision Transformers. Big thanks to [Christian Liebhardt](https://github.com/liebharc) (for open-sourcing homr).

homr itself uses 2 differnent models: A segmentation model and a transformer model (encoder and decoder).
While the segmentation model and the transformer encoder use LiteRT as a backend, the transformer decoder uses OnnxRuntime which works well with dynamic shapes.

The model inference is using native Java APIs accessed via pyjnius. On most recent phones you should be able to transform one page in around one minute.

## Build
*Note: Buildozer is really fragile and support by me is limited.*

Andromr is built with buildozer under WSL. If you are new to buildozer you can follow this [tutorial](https://www.youtube.com/watch?v=pzsvN3fuBA0) for setting buildozer up.

Modifications to buildozer are listed in `buildozer.spec`.

## Contributing
PRs are welcome :)

If you only want to change something about the homr backend please first think about contributing to homr. Sooner or later, all the changes made to homr will be added to Andromr too.

## Acknowledgments
Thanks to [Christian Liebhardt](https://github.com/liebharc) for open-sourcing homr.
All the other open-source licenses are listed in `oss_licenses.txt`.

## Future Plans
- Release of Andromr in PlayStore
- Improve Decoder performance

## License
Andromr is open-sourced under the AGPL 3.0 license.
