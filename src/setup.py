from setuptools import setup, find_packages

setup(
    name='agentware',
    version='0.1.6',
    description='A framework that builds agents with short-term memory management, longterm management',
    long_description='long description',
    author='darthjaja',
    author_email='your-email@example.com',
    packages=find_packages(),
    package_data={
        'agentware': ['base_agent_configs/*.json']
    },
    install_requires=[
        "tiktoken==0.3.3",
        "openai==0.27.2",
        "jwt==1.3.1",
        "colorlog==6.7.0",
        "python-dotenv==1.0.0",
        "pystache==0.6.0"
    ],
)
