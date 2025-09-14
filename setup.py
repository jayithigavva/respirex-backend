from setuptools import setup, find_packages

setup(
    name="respirex-backend",
    version="1.0.0",
    description="AI-powered respiratory disease detection backend",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.100.0",
        "uvicorn==0.23.0",
        "python-multipart==0.0.6",
        "librosa==0.10.0",
        "soundfile==0.12.1",
        "numpy==1.24.0",
        "scipy==1.10.0",
        "pandas==2.0.0",
        "scikit-learn==1.3.0",
        "joblib==1.3.0",
        "pydantic==2.0.0",
    ],
    python_requires=">=3.10",
)
