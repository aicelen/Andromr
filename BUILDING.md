Andromr is built using the default buildozer docker image. Although this gives some consitency, buildozer remains fragile. The following steps generate a debug apk.

#### 1. Install buildozer docker image:
- Follow the instructions in buildozers readme


#### 2. Modify the following files:
- Add ```ndk {abiFilters 'arm64-v8a'}``` under defaultconfig to:
```
andromr/.buildozer/android/platform/python-for-android/pythonforandroid/bootstraps/common/build/templates/build.tmpl.gradle
```
- Add ```android:enableOnBackInvokedCallback="false"``` under application to:
```
andromr/.buildozer/android/platform/python-for-android/pythonforandroid/bootstraps/sdl_common/build/templates/build.tmpl.gradle
```

#### 3. Build command:
```
sudo docker run --interactive --tty --rm     --volume "$HOME/.buildozer":/home/user/.buildozer     --volume "$PWD":/home/user/hostcwd     kivy/buildozer android debug
```

#### Other useful commands
- Make a clean build
```
sudo docker run --interactive --tty --rm     --volume "$HOME/.buildozer":/home/user/.buildozer     --volume "$PWD":/home/user/hostcwd     kivy/buildozer android clean
```

- Use logcat filtering for debugging
```
adb logcat *:S python:D
```
