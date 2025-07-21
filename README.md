# Andromr

Optical Music Recognition (OMR) for Android using [oemer](https://github.com/BreezeWhite/oemer).

## Technical Details
Andromr uses **Kivy** and **KivyMD** for the UI, with **Buildozer** to generate the APK. The OMR engine is based on [Oemer](https://github.com/BreezeWhite/oemer), which produces strong recognition results.

-   Oemer’s CNN models are converted to TFLite using dynamic integer quantization.
-   Inference is handled via LiteRT, using [teticio’s example](https://github.com/teticio/kivy-tensorflow-helloworld).
-   Currently, inference runs on a single thread using XNNPACK, resulting in modest performance (~4 minutes per page on Exynos 1280).

## Build
*Note: Buildozer is really fragile and support by me is limited.*

Andromr is built with buildozer under WSL. If you are new to buildozer you can follow this [tutorial](https://www.youtube.com/watch?v=pzsvN3fuBA0) for setting buildozer up.

#### Setting up OpenCV:
1.  Download the `opencv-android-sdk` (version 4.7.0) from [opencv.org](https://opencv.org/)
3.  Copy the `native` folder from the SDK into a new folder named `opencv`, placed at the same level as your `buildozer.spec` file.

## Planned Features

 - LiteRT Next including GPU support (already done but waiting for the fix of two vulnerabilities)
 - mulit page support
 - general performance improvemets concerning the app

## Acknowledgments
Thanks to [BreezeWhite](https://github.com/BreezeWhite) for open-sourcing Oemer. 
Open-source licenses for dependencies are listed in `data/licenses.txt`.


## License
Andromr is currently All Rights Reserved. 
However it's opencode:

 - You are allowed to read the code and build the APK for personal use
 - Distribution is not allowed without explicit permission

This approach ensures user safety (downloading a random apk is risky) and supports the Kivy community by contributing a more complex open project.

*If you want to use parts of Andromr in your own project, please contact me.*