"""
Setup configuration for SoilVisioNet Production System.
Enables installation as a Python package and dependency management.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="soilvisionet-production",
    version="1.0.0",
    author="SoilVisioNet Team",
    description="Professional dual-mode agricultural intelligence system for crop disease detection and soil assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/soilvisionet/production",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Agricultural Science",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "transformers>=4.30.2",
        "streamlit>=1.26.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0.76",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.2",
            "seaborn>=0.12.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "soilvisionet=ui.app:main",
        ],
    },
    project_urls={
        "Documentation": "https://github.com/soilvisionet/production/docs",
        "Source Code": "https://github.com/soilvisionet/production",
        "Issue Tracker": "https://github.com/soilvisionet/production/issues",
    },
    keywords=[
        "agriculture",
        "crop disease detection",
        "computer vision",
        "vision transformer",
        "soil assessment",
        "deep learning",
        "machine learning",
        "farming",
        "agronomy",
    ],
    zip_safe=False,
)
