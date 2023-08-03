from setuptools import setup, find_packages

setup(
    name='agentware',
    version='0.1.5',
    description='A framework that builds agents with short-term memory management, longterm management',
    long_description='long description',
    author='darthjaja',
    author_email='your-email@example.com',
    packages=find_packages(),
    package_data={
        'agentware': ['base_agent_configs/*.json']
    },
    install_requires=[
        "tiktoken",
        "openai",
        "jwt",
        "colorlog",
        "python-dotenv"
    ],
)
