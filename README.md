# Andromr

<a href='https://play.google.com/store/apps/details?id=com.aicelen.andromr'><img alt='Get it on Google Play' src='https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png' height=71/></a>

Optical Music Recognition (convert a picture of music notes to a machine readable format like .musicxml) for Android using [homr](https://github.com/liebharc/homr).

## How to use
- Scan or open image(s)/pdf
- Run optical music recognition (leave the app focused)
- Export via the Share button

## Features
- All features of homr (near state-of-the-art optical music recognition)
- Support for multiple pages
- Privacy (your scores stay on your device)

## Technical Details
Andromr uses **Kivy** and **KivyMD** for the UI, with **Buildozer** to generate the APK. The OMR engine is based on [homr](https://github.com/liebharc/homr), which provides great quality due to the use of Machine Learning. Big thanks to [Christian Liebhardt](https://github.com/liebharc) (for open-sourcing homr).

Homr itself uses 2 differnent models: A segmentation model and a transformer model (encoder and decoder).
While the segmentation model and the transformer encoder run in LiteRT, the transformer decoder uses OnnxRuntime.

The inference code is written in Java and called from the python app using pyjnius. On most recent phones you should be able to scan one page in around 30 seconds.

## Build
See BUILDING.md

## Contributing
PRs are welcome :)

If you only want to change something about the homr backend please first think about contributing to homr. Sooner or later, all the changes made to homr will be added to Andromr too.

## Acknowledgments
Thanks to [Christian Liebhardt](https://github.com/liebharc) for open-sourcing homr.
All open-source licenses are listed in `oss_licenses.txt`.

## License
Andromr is open-sourced under the AGPL 3.0 license.
