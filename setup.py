from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys

def read_version_number():
    with open('version_number.txt', 'r') as file:
        version_number = file.readline()
    return version_number.strip()


def find_cmake():
    try:
        subprocess.check_output(['cmake', '--version'])
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("CMake must be installed to build negative_edge_sampler")


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        find_cmake()
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Check for debug mode
        is_debug = os.environ.get('DEBUG_BUILD', '0').lower() in ('1', 'true', 'yes')
        build_type = 'Debug' if is_debug else 'Release'

        print(f"Building in {build_type} mode")

        # CMake arguments
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_BUILD_TYPE={build_type}',
        ]

        # Use environment-defined Python paths if available
        python_executable = os.environ.get('PYTHON_EXECUTABLE', sys.executable)
        python_include_dir = os.environ.get('PYTHON_INCLUDE_DIR', '')
        python_library = os.environ.get('PYTHON_LIBRARY', '')

        cmake_args.append(f'-DPYTHON_EXECUTABLE={python_executable}')

        if python_include_dir:
            cmake_args.append(f'-DPYTHON_INCLUDE_DIR={python_include_dir}')

        if python_library:
            cmake_args.append(f'-DPYTHON_LIBRARY={python_library}')

        # Debug output
        print(f"Building with Python: {python_executable}")
        if python_include_dir:
            print(f"Python include dir: {python_include_dir}")
        if python_library:
            print(f"Python library: {python_library}")

        build_args = ['--config', build_type]
        os.makedirs(self.build_temp, exist_ok=True)

        try:
            env = os.environ.copy()
            subprocess.check_call(
                ['cmake', os.path.abspath('.')] + cmake_args,
                cwd=self.build_temp,
                env=env
            )
            subprocess.check_call(
                ['cmake', '--build', '.'] + build_args,
                cwd=self.build_temp,
                env=env
            )
        except subprocess.CalledProcessError as e:
            print(f"Error during CMake configuration or build: {e}")
            sys.exit(1)


setup(
    name="temporal-negative_edge-sampler",
    version=read_version_number(),
    author="Ashfaq Salehin",
    author_email="ashfaq.salehin1701@gmail.com",
    description="Fast temporal negative edge sampler for graph neural networks",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ashfaq1701/temporal-negative-edge-sampler",
    ext_modules=[CMakeExtension('temporal_negative_edge_sampler')],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["numpy>=1.19.0"],
)