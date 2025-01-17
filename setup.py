from setuptools import setup, find_packages

setup(
    name='edd_utils',  
    version='0.1.0',  
    description='A brief description of your framework',  
    author='Erland Hilman Fuadi',  
    author_email='erland.pg366@gmail.com',  
    url='https://github.com/Erland366/edd_utils',  
    packages=find_packages(where="src"),  
    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',  
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.7',  
    package_dir={"" "src"},
)