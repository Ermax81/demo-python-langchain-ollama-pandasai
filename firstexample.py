# sources:
#
# https://medium.com/@amadatiq/pandasai-making-data-analysis-conversational-and-fun-3acc76584cb3

from pandasai import SmartDataframe
import pandas as pd
from langchain_community.llms import Ollama

llm = Ollama(model="llama3:8b")

df = pd.DataFrame({
    "country": [
        "United States", "United Kingdom", "France", "Germany", "Italy",
        "Spain", "Canada", "Australia", "Japan", "China",
    ],
    "gdp": [
        19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416,
        1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064,
    ],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12],
})

sdf = SmartDataframe(df, config={"llm": llm})

print( sdf.chat('Which are the 5 happiest countries?'))
#{'type': 'dataframe', 'value':           country             gdp  happiness_index
# 6          Canada   1607402389504             7.23
# 7       Australia   1490967855104             7.22
# 1  United Kingdom   2891615567872             7.16
#     3         Germany   3435817336832             7.07
#     0   United States  19294482071552             6.94}
#country             gdp  happiness_index
#6          Canada   1607402389504             7.23
#7       Australia   1490967855104             7.22
#1  United Kingdom   2891615567872             7.16
#3         Germany   3435817336832             7.07
#0   United States  19294482071552             6.94

print( sdf.chat('What is the GDP of the United States?'))
#{'type': 'string', 'value': 'The GDP of United States is 19294482071552.'}
#The GDP of United States is 19294482071552.
#The happiest country is Canada.

print( sdf.chat('What is the sum of the GDPs of the 2 happiest countries?'))
#{'type': 'string', 'value': 'The sum of the GDPs of the 2 happiest countries is 38438935724032.'}
#The sum of the GDPs of the 2 happiest countries is 38438935724032.