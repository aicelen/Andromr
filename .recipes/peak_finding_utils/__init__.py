from pythonforandroid.recipe import CythonRecipe


class Peak_finding_utils(CythonRecipe):
    version = "v0.1"
    url = "https://github.com/aicelen/peak-finding-utils/archive/refs/tags/v0.1.zip"
    name = "peak-finding-utils"

    depends = ["python3", "setuptools", "numpy"]


recipe = Peak_finding_utils()
