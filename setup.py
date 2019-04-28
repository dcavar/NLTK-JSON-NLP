
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nltkjsonnlp",
    version="0.0.3",
    author="Damir Cavar, Oren Baldinger, Maanvitha Gongalla, Anurag Kumar, Murali Kammili, Boli Fang",
    author_email="damir@cavar.me",
    description="The Python NLTK JSON-NLP package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dcavar/NLTK-JSON-NLP",
    packages=setuptools.find_packages(),
    install_requires=[
        'nltk>=3.4',
        'pyjsonnlp>=0.2.9'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    setup_requires=["pytest-runner"],
    test_suite="tests",
    tests_require=["pytest", "coverage"]
)
