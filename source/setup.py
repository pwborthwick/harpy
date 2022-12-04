from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules=[Extension("aello",["aello.pyx"],libraries=["m"],extra_compile_args=["-ffast-math"])]
ext_modules=[Extension("ocypete",["ocypete.pyx"],libraries=["m"],extra_compile_args=["-ffast-math"])]
setup(name='aello',ext_modules = cythonize('aello.pyx',language_level=3))
setup(name='ocypete',ext_modules = cythonize('ocypete.pyx',language_level=3))

