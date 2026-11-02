from setuptools import setup, find_packages

setup(
    name="linecount",
    version="0.1.0",
    author="Habimana Moise",
    author_email="habimanamoise77@gmail.com",
    description="Real-Time Industrial Object Detection, Tracking and Counting",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/moise744/linecount-ai",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
        "PyYAML>=6.0",
    ],
    extras_require={
        "app": ["streamlit>=1.28.0"],
        "dev": ["pytest>=7.4.0", "httpx>=0.24.0"],
    },
)
