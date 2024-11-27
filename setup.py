from setuptools import find_packages, setup


def get_requires():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        file_content = f.read()
        lines = [
            line.strip()
            for line in file_content.strip().split("\n")
            if not line.startswith("#")
        ]
        return lines


setup(
    name="zigent",
    version="0.0.1",
    url="https://github.com/hu-qi/zigent",
    description="AI Agent for zishu.co",
    packages=find_packages(exclude=["test*", "app*", "doc*", "example"]),
    python_requires=">=3.9",
    install_requires=get_requires(),
    license="Apache License 2.0",
    author="Zhiwei Liu, Qi Hu",
    author_email="zhiweiliu@salesforce.com",
)
