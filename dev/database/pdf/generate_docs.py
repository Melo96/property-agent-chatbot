import json
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

from dotenv import load_dotenv
load_dotenv()

pdf_path = "/Users/yangkaiwen/Documents/hypergai-chatbot-data/handbook_demo/adobe_handbook.pdf"
output_path = Path('/Users/yangkaiwen/Documents/hypergai-chatbot-data/handbook_demo/docs')
# Returns a List[Element] present in the pages of the parsed pdf document
elements = partition_pdf("/Users/yangkaiwen/Documents/hypergai-chatbot-data/handbook_demo/adobe_handbook.pdf")
elements = [element.to_dict() for element in elements]

