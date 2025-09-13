from pythonforandroid.recipe import CythonRecipe

class Ni_label(CythonRecipe):
    version = 'v0.1'
    url = 'https://github.com/aicelen/ni_label/archive/refs/tags/v0.1.zip'
    name = 'ni_label'

    depends = ['python3', 'setuptools', 'numpy']

recipe = Ni_label()
